#!/usr/bin/env python3
"""
V5: ROI指定・広域スケール探索版
1. ユーザーがInput画像から「探したい屋根」をマウスで囲む（ROI指定）。
2. その切り抜き画像をテンプレートとして、Reference画像内を広範囲のスケールで探索する。
"""
import sys
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(input_path, reference_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    return cv2.imread(input_path), cv2.imread(reference_path)

def get_edges(img, is_template=False):
    """エッジ検出（テンプレートの場合は内部を塗りつぶして輪郭重視にするなどの処理が可能）"""
    # ブラーでノイズ除去
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Cannyエッジ検出
    edges = cv2.Canny(gray, 50, 150)
    return edges

def rotate_and_scale(img, angle, scale):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 画像サイズ拡張
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated, new_w, new_h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.jpg')
    parser.add_argument('--reference', default='reference.jpg')
    parser.add_argument('--tl-lat', type=float, default=35.0)
    parser.add_argument('--tl-lon', type=float, default=135.0)
    parser.add_argument('--br-lat', type=float, default=34.9)
    parser.add_argument('--br-lon', type=float, default=135.1)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    # 1. 画像読み込み
    img_inp, img_ref = load_images(args.input, args.reference)
    ref_h, ref_w = img_ref.shape[:2]

    print("=======================================================")
    print("ステップ1: 検索対象の指定")
    print("表示されるウィンドウで、Input画像の中の「屋根」の部分だけを")
    print("マウスドラッグで囲んで、[ENTER] または [SPACE] を押してください。")
    print("※壁や空、隣の家が入らないように、屋根の形状だけを囲むのがコツです。")
    print("=======================================================")

    # 2. ROI選択（Input画像からテンプレートを作成）
    # ウィンドウサイズが大きすぎる場合はリサイズして表示
    disp_h, disp_w = img_inp.shape[:2]
    scale_disp = 1.0
    if disp_w > 1200:
        scale_disp = 1200 / disp_w
        img_inp_disp = cv2.resize(img_inp, None, fx=scale_disp, fy=scale_disp)
    else:
        img_inp_disp = img_inp

    roi = cv2.selectROI("Select Roof", img_inp_disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Roof")

    if roi[2] == 0 or roi[3] == 0:
        print("範囲が選択されませんでした。終了します。")
        sys.exit(1)

    # 選択範囲を元のスケールに戻して切り抜き
    x, y, w, h = [int(v / scale_disp) for v in roi]
    template_color = img_inp[y:y+h, x:x+w]
    
    # 3. エッジ抽出
    # Input(テンプレート)のエッジ
    edges_template = get_edges(template_color, is_template=True)
    # Reference(全体地図)のエッジ
    edges_ref = get_edges(img_ref)

    print("\nステップ2: 全探索開始")
    print("Reference画像内で一致する場所を探しています...")
    print("スケール範囲: 0.05倍(極小) ～ 0.50倍(中)")

    best_val = -1
    best_loc = (0, 0)
    best_params = {}
    best_match_vis = None

    # 探索パラメータ（ここを修正：かなり小さいサイズから探すように変更）
    # 0.05 (5%) から 0.5 (50%) まで
    scales = np.arange(0.05, 0.51, 0.03) 
    angles = range(0, 360, 10) # 10度刻み

    total_steps = len(scales) * len(angles)
    step = 0

    for scale in scales:
        for angle in angles:
            step += 1
            if step % 200 == 0:
                print(f"進捗: {step}/{total_steps} (Best: {best_val:.3f})")

            # テンプレートを変形
            t_img, t_w, t_h = rotate_and_scale(edges_template, angle, scale)

            # テンプレートが大きすぎる、または小さすぎる場合はスキップ
            if t_w >= ref_w or t_h >= ref_h or t_w < 10 or t_h < 10:
                continue

            # テンプレートマッチング
            res = cv2.matchTemplate(edges_ref, t_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_params = {'angle': angle, 'scale': scale, 'w': t_w, 'h': t_h}
                # 可視化用に保持
                best_match_vis = t_img

    if best_val < 0.15: # 閾値
        print(f"\n失敗: 一致する箇所が見つかりませんでした (Max Score: {best_val:.3f})")
        print("切り抜き範囲を変えて再度試すか、手動特定を行ってください。")
        sys.exit(1)

    print(f"\n★ 発見しました！ (Score: {best_val:.3f})")
    print(f"パラメータ: 回転 {best_params['angle']}度, スケール {best_params['scale']:.2f}倍")

    # 中心座標
    top_left = best_loc
    w_match = best_params['w']
    h_match = best_params['h']
    center_x = top_left[0] + w_match // 2
    center_y = top_left[1] + h_match // 2

    # 緯度経度変換
    lat = args.tl_lat + (center_y / ref_h) * (args.br_lat - args.tl_lat)
    lon = args.tl_lon + (center_x / ref_w) * (args.br_lon - args.tl_lon)

    print(f"Reference座標: X={center_x}, Y={center_y}")
    print(f"推定緯度: {lat:.7f}")
    print(f"推定経度: {lon:.7f}")

    if not args.no_show:
        res_img = img_ref.copy()
        cv2.rectangle(res_img, top_left, (top_left[0]+w_match, top_left[1]+h_match), (0, 0, 255), 3)
        cv2.circle(res_img, (center_x, center_y), 5, (0, 255, 0), -1)

        plt.figure(figsize=(12, 6))
        
        # 左：入力画像の切り抜き
        plt.subplot(1, 3, 1)
        plt.title("Selected Template")
        plt.imshow(cv2.cvtColor(template_color, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 中：マッチした変形エッジ
        plt.subplot(1, 3, 2)
        plt.title(f"Matched Shape\n(Sc:{best_params['scale']:.2f}, Rot:{best_params['angle']})")
        plt.imshow(best_match_vis, cmap='gray')
        plt.axis('off')

        # 右：全体の結果
        plt.subplot(1, 3, 3)
        plt.title("Result on Map")
        plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()