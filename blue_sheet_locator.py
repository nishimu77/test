#!/usr/bin/env python3
"""
台風被害の航空写真（input.jpg）と衛星写真（reference.jpg）をマッチングし、
航空写真内で検出したブルーシートの緯度経度を求めるスクリプト。

使い方の例:
python blue_sheet_locator.py \
  --input input.jpg --reference reference.jpg \
  --tl-lat 35.000 --tl-lon 135.000 --br-lat 34.900 --br-lon 135.100

必要ライブラリ: opencv-python, numpy, matplotlib
"""
import sys
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images(input_path, reference_path):
    """画像をカラーとグレースケールで読み込む。存在しない場合は例外を投げる。"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    img_input_color = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img_ref_color = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    if img_input_color is None:
        raise IOError(f"Failed to load input image: {input_path}")
    if img_ref_color is None:
        raise IOError(f"Failed to load reference image: {reference_path}")

    img_input_gray = cv2.cvtColor(img_input_color, cv2.COLOR_BGR2GRAY)
    img_ref_gray = cv2.cvtColor(img_ref_color, cv2.COLOR_BGR2GRAY)
    return img_input_color, img_input_gray, img_ref_color, img_ref_gray


def match_keypoints_sift(img1_gray, img2_gray, min_matches=10):
    """SIFT特徴点抽出とFLANNでのマッチング、ホモグラフィ推定を行う。
    成功すると (matches_mask, H, kp1, kp2, good_matches) を返す。
    マッチが不十分なら H は None を返す。
    """
    try:
        # より多くの特徴点を検出するためパラメータ調整
        sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.03, edgeThreshold=15)
    except Exception:
        # 古いOpenCVでは xfeatures2d にある場合がある
        try:
            sift = cv2.xfeatures2d.SIFT_create()
        except Exception as e:
            raise RuntimeError("SIFTが利用できません。OpenCV-contribが必要です。") from e

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None, None, kp1, kp2, []

    # FLANNパラメータ（KDTree）
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Loweの比率テストで良いマッチを選別
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < min_matches:
        return None, None, kp1, kp2, good

    # マッチ点を配列に変換
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist() if mask is not None else None
    return matches_mask, H, kp1, kp2, good


def match_with_multiscale(img1_gray, img2_gray, min_matches=10, scale_factors=None):
    """マルチスケールで画像マッチングを試行する。
    img1をさまざまなスケールで縮小してimg2とマッチングし、最良の結果を返す。
    クローズアップ画像と広域画像の間のスケール差に対応するため。
    
    Returns:
        best_result: (matches_mask, H_adjusted, kp1, kp2, good, scale) または (None, None, None, None, [], 1.0)
    """
    if scale_factors is None:
        # デフォルトのスケール係数（1.0から0.05まで、広範囲をカバー）
        scale_factors = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
    
    best_result = (None, None, None, None, [], 1.0)
    best_match_count = 0
    
    for scale in scale_factors:
        if scale != 1.0:
            h, w = img1_gray.shape
            new_w = max(int(w * scale), 20)
            new_h = max(int(h * scale), 20)
            img1_scaled = cv2.resize(img1_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img1_scaled = img1_gray
        
        matches_mask, H, kp1, kp2, good = match_keypoints_sift(img1_scaled, img2_gray, min_matches=min_matches)
        
        if H is not None and len(good) > best_match_count:
            # スケールを補正したホモグラフィ行列を作成
            if scale != 1.0:
                # スケール行列: 元の座標をスケール座標に変換
                S = np.array([[scale, 0, 0],
                              [0, scale, 0],
                              [0, 0, 1]], dtype=np.float32)
                S_inv = np.array([[1/scale, 0, 0],
                                  [0, 1/scale, 0],
                                  [0, 0, 1]], dtype=np.float32)
                # H_adjusted = H @ S (元画像座標 -> スケール画像座標 -> ref座標)
                H_adjusted = H @ S
            else:
                H_adjusted = H
            
            best_result = (matches_mask, H_adjusted, kp1, kp2, good, scale)
            best_match_count = len(good)
            print(f"  スケール {scale:.2f} で {len(good)} マッチを検出")
    
    if best_result[1] is not None:
        print(f"  → 最良スケール: {best_result[5]:.2f} ({best_match_count} マッチ)")
    
    return best_result


def match_patch_to_reference(patch_gray, ref_gray, min_matches=6):
    """入力画像から切り出したパッチを参照画像にマッチングするためのラッパ。
    マルチスケールでマッチングを試行する。
    成功すると (matches_mask, H, kp_patch, kp_ref, good_matches) を返す。
    """
    result = match_with_multiscale(patch_gray, ref_gray, min_matches=min_matches)
    # scale情報は除いて返す（互換性のため）
    return result[0], result[1], result[2], result[3], result[4]


def detect_blue_sheets(img_color):
    """HSV閾値で青色領域を抽出し、輪郭の中心座標を返す（ピクセル座標）。"""
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    # 青色のHSV範囲（調整が必要な場合あり）
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # ノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # 小さすぎる領域は除外（調整可）
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return centers, mask, contours


def transform_points(pts, H):
    """ホモグラフィで入力画像上の点群を参照画像座標系に変換する。"""
    if H is None:
        raise ValueError("ホモグラフィが無いため変換できません。")
    if len(pts) == 0:
        return []
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(arr, H)
    dst = dst.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in dst]


def pixel_to_latlon(x, y, ref_w, ref_h, tl_lat, tl_lon, br_lat, br_lon):
    """参照画像のピクセル座標(x,y)を緯度経度に線形補間で変換する。"""
    # 線形補間: y によって緯度、x によって経度を補間
    lat = tl_lat + (y / ref_h) * (br_lat - tl_lat)
    lon = tl_lon + (x / ref_w) * (br_lon - tl_lon)
    return lat, lon


def draw_visualization(img_ref_color, transformed_corners, transformed_centers):
    """参照画像上に入力画像の枠と検出点を描画して表示する。"""
    vis = img_ref_color.copy()
    # 入力画像の枠（単一または複数のポリゴン）をポリラインで描画
    if transformed_corners is not None:
        # 単一のポリゴン(4点)が来た場合
        if isinstance(transformed_corners, (list, tuple)) and len(transformed_corners) > 0 and isinstance(transformed_corners[0], (list, tuple)):
            # もし最初要素が座標タプルなら transformed_corners は単一ポリゴン
            if len(transformed_corners) == 4 and not any(isinstance(p[0], (list, tuple)) for p in transformed_corners):
                pts = np.int32(transformed_corners).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            else:
                # 複数ポリゴンのとき
                for poly in transformed_corners:
                    if poly is None:
                        continue
                    try:
                        pts = np.int32(poly).reshape((-1, 1, 2))
                        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                    except Exception:
                        continue

    # 検出点を描画
    for (x, y) in transformed_centers:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)

    # BGR->RGB for matplotlib
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_rgb)
    plt.axis('off')
    plt.title('Reference image with matched input frame and detected blue sheets')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Blue sheet geo-locator')
    parser.add_argument('--input', default='input.jpg', help='航空写真（位置情報なし）')
    parser.add_argument('--reference', default='reference.jpg', help='衛星写真（位置情報あり）')
    parser.add_argument('--tl-lat', type=float, required=True, help='Reference画像左上の緯度')
    parser.add_argument('--tl-lon', type=float, required=True, help='Reference画像左上の経度')
    parser.add_argument('--br-lat', type=float, required=True, help='Reference画像右下の緯度')
    parser.add_argument('--br-lon', type=float, required=True, help='Reference画像右下の経度')
    parser.add_argument('--min-matches', type=int, default=10, help='ホモグラフィ推定に必要な最小マッチ数')
    parser.add_argument('--no-show', action='store_true', help='可視化を表示しない')

    args = parser.parse_args()

    try:
        inp_color, inp_gray, ref_color, ref_gray = load_images(args.input, args.reference)
    except Exception as e:
        print(f"画像読み込みエラー: {e}")
        sys.exit(1)

    # まず入力画像からブルーシート領域を検出して、その領域ごとに
    # 参照画像へ切り出しパッチでマッチングを試みる。
    centers, mask, contours = detect_blue_sheets(inp_color)
    transformed_centers = []
    transformed_polygons = []

    if len(centers) == 0:
        print("入力画像からブルーシートと思われる領域が検出されませんでした。全体マッチにフォールバックします。")
    else:
        print(f"検出されたブルーシート候補数: {len(contours)}。各候補を参照画像にマッチングします...")
        # 各輪郭についてバウンディングボックスでパッチを切り出してマッチング
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # パッチに余裕を持たせるマージン（周辺の特徴を多く含めるため大きめに取る）
            margin = int(max(w, h) * 1.5)  # 0.2から1.5に大幅増
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(inp_color.shape[1], x + w + margin)
            y1 = min(inp_color.shape[0], y + h + margin)
            patch_color = inp_color[y0:y1, x0:x1]
            patch_gray = inp_gray[y0:y1, x0:x1]

            if patch_gray.size == 0:
                continue

            print(f"候補パッチをマッチング: bbox=({x0},{y0},{x1-x0},{y1-y0})")
            matches_mask_p, H_p, kp_p, kp_r, good_p = match_patch_to_reference(patch_gray, ref_gray, min_matches=6)
            if H_p is None:
                print(" -> この候補は参照画像にマッチしませんでした。")
                continue

            print(" -> この候補を参照画像にマッチしました。")
            # 輪郭の中心をpatch座標系に変換して参照座標系へ
            M = cv2.moments(cnt)
            if M.get("m00", 0) == 0:
                continue
            cx = int(M["m10"] / M["m00"]) - x0
            cy = int(M["m01"] / M["m00"]) - y0
            try:
                tcent = transform_points([(cx, cy)], H_p)[0]
                transformed_centers.append(tcent)
            except Exception as e:
                print(f"候補点の変換に失敗: {e}")

            # パッチの四隅を参照座標系へ変換して可視化用ポリゴンを得る
            ph, pw = patch_gray.shape
            p_corners = [(0, 0), (pw - 1, 0), (pw - 1, ph - 1), (0, ph - 1)]
            try:
                tpoly = transform_points(p_corners, H_p)
                transformed_polygons.append(tpoly)
            except Exception:
                transformed_polygons.append(None)

    # もしいずれの候補も参照画像上にマッチしなかった場合は、従来通り全体マッチにフォールバック
    if len(transformed_centers) == 0:
        print("パッチマッチで位置を特定できませんでした。全体画像でマルチスケールマッチングを試みます...")
        matches_mask, H, kp1, kp2, good, scale = match_with_multiscale(inp_gray, ref_gray, min_matches=args.min_matches)
        if H is None:
            print("十分な良好なマッチが得られず、ホモグラフィを推定できませんでした。処理を中止します。")
            sys.exit(1)
        print(f"全体マッチでホモグラフィを推定しました（スケール: {scale:.2f}）。")

        # 全体ホモグラフィを用いて検出中心を変換
        try:
            transformed_centers = transform_points(centers, H) if len(centers) > 0 else []
        except Exception as e:
            print(f"点変換エラー: {e}")
            transformed_centers = []

        # 入力画像の4隅を変換して参照画像上の枠を得る
        h_inp, w_inp = inp_gray.shape
        corners = [(0, 0), (w_inp - 1, 0), (w_inp - 1, h_inp - 1), (0, h_inp - 1)]
        try:
            transformed_corners = transform_points(corners, H)
        except Exception:
            transformed_corners = None
    else:
        # パッチマッチで得た複数のポリゴンを可視化用に使う
        transformed_corners = transformed_polygons

    # 参照画像の解像度
    ref_h, ref_w = ref_gray.shape

    # ピクセルを緯度経度に変換
    latlon_list = []
    for (x, y) in transformed_centers:
        lat, lon = pixel_to_latlon(x, y, ref_w, ref_h, args.tl_lat, args.tl_lon, args.br_lat, args.br_lon)
        latlon_list.append((lat, lon))

    # 結果出力
    if len(latlon_list) > 0:
        print("検出されたブルーシートの緯度・経度一覧:")
        for i, (lat, lon) in enumerate(latlon_list, start=1):
            print(f"{i}: lat={lat:.7f}, lon={lon:.7f}")
    else:
        print("緯度経度は出力されませんでした。")

    # 可視化
    if not args.no_show:
        draw_visualization(ref_color, transformed_corners, transformed_centers)


if __name__ == '__main__':
    main()