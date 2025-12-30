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
        sift = cv2.SIFT_create()
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
    # 入力画像の枠をポリラインで描画
    if transformed_corners is not None and len(transformed_corners) == 4:
        pts = np.int32(transformed_corners).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

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

    print("特徴点マッチング（SIFT）を実行しています...")
    matches_mask, H, kp1, kp2, good = match_keypoints_sift(inp_gray, ref_gray, min_matches=args.min_matches)

    if H is None:
        print("十分な良好なマッチが得られず、ホモグラフィを推定できませんでした。処理を中止します。")
        sys.exit(1)

    print("ホモグラフィを推定しました。")

    # ブルーシート検出
    centers, mask, contours = detect_blue_sheets(inp_color)
    if len(centers) == 0:
        print("入力画像からブルーシートと思われる領域が検出されませんでした。")
    else:
        print(f"検出されたブルーシート中心点数: {len(centers)}")

    # 中心を参照画像座標系に変換
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
