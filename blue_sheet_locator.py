#!/usr/bin/env python3
"""
V9: コンテキスト（文脈）探索 & Top-5 候補表示版
【重要】
ROI選択の際は、探したい家単体ではなく、「隣の家」や「前の道路」などを含めた
「特徴的な並び（パターン）」として大きく囲ってください。
単体の形状あてよりも圧倒的に精度が向上します。
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

def create_non_blue_mask(img):
    """ブルーシート（青色）部分を特定するマスクを作成"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask

def get_edges(img, is_template=False):
    # テンプレート（入力画像）の前処理：ブルーシートを消す（インペイント）
    if is_template:
        mask = create_non_blue_mask(img)
        # 青い部分があれば、周囲の色で埋める（シワによるノイズを消して輪郭だけ残すため）
        if cv2.countNonZero(mask) > 0:
            img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # ノイズ除去
    k_size = (5, 5) if is_template else (7, 7)
    blurred = cv2.GaussianBlur(img, k_size, 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出
    # Reference側は畑などのノイズを拾わないよう閾値を高めに
    t1 = 50 if is_template else 100
    t2 = 150 if is_template else 200
    edges = cv2.Canny(gray, t1, t2)
    
    # ★重要: 線を太くする（テンプレートのみ）
    # 縮小しても家の形が消えないようにする。Reference側は太くしない（くっつき防止）
    if is_template:
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
    return edges

def rotate_and_scale(img, angle, scale):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated, new_w, new_h

def non_max_suppression(candidates, min_dist=30, top_k=5):
    """
    候補の中から、距離が近い（重複している）ものを除外し、
    スコアが高い順にTop-K個を選ぶ
    """
    # スコア順にソート（降順）
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    final_picks = []
    for cand in candidates:
        if len(final_picks) >= top_k:
            break
            
        is_far_enough = True
        cand_pt = np.array(cand['pt'])
        
        for pick in final_picks:
            pick_pt = np.array(pick['pt'])
            dist = np.linalg.norm(cand_pt - pick_pt)
            if dist < min_dist: # 近くに既に選ばれた候補がある場合はスキップ
                is_far_enough = False
                break
        
        if is_far_enough:
            final_picks.append(cand)
            
    return final_picks

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

    # 画像読み込み
    img_inp, img_ref = load_images(args.input, args.reference)
    ref_h, ref_w = img_ref.shape[:2]

    # --- ROI選択 ---
    print("=======================================================")
    print("【ステップ1】検索パターンの指定")
    print("ウィンドウが開いたら、探したい家を囲みますが...")
    print("★重要★ 家単体ではなく、「隣の家」や「道路」も含めて")
    print("　　　　 『3〜4軒ぶんのエリア』を大きく囲ってください！")
    print("　　　　 特徴的な並びパターンを作ることで精度が劇的に上がります。")
    print("=======================================================\n")
    print ("test")
    
    
    disp_h, disp_w = img_inp.shape[:2]
    scale_disp = 1.0
    if disp_w > 1200:
        scale_disp = 1200 / disp_w
        img_inp_disp = cv2.resize(img_inp, None, fx=scale_disp, fy=scale_disp)
    else:
        img_inp_disp = img_inp

    roi = cv2.selectROI("Select Context (House + Neighbors)", img_inp_disp, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        print("範囲未選択のため終了します。")
        sys.exit(1)

    x, y, w, h = [int(v / scale_disp) for v in roi]
    template_color = img_inp[y:y+h, x:x+w]
    
    # エッジ抽出（ブルーシート除去＆太線化含む）
    edges_template = get_edges(template_color, is_template=True)
    edges_ref = get_edges(img_ref)

    # --- 荒探索 (Coarse Search) ---
    print("\n【ステップ2】広域スキャン開始 (Top-5 候補を探索)")
    
    # 探索パラメータ
    # 0.04倍(極小)から0.40倍まで、少し細かくスキャン
    scales = np.arange(0.04, 0.41, 0.02) 
    angles = range(0, 360, 10) 

    candidates = []
    step = 0
    total_steps = len(scales) * len(angles)

    for scale in scales:
        for angle in angles:
            step += 1
            if step % 1000 == 0: print(f"スキャン進捗: {int(step/total_steps*100)}%")

            t_img, t_w, t_h = rotate_and_scale(edges_template, angle, scale)
            if t_w >= ref_w or t_h >= ref_h or t_w < 10 or t_h < 10: continue

            res = cv2.matchTemplate(edges_ref, t_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # ある程度の一致（0.15以上）があれば候補として保存
            if max_val > 0.15:
                candidates.append({
                    'score': max_val,
                    'pt': max_loc,
                    'angle': angle,
                    'scale': scale,
                    'w': t_w, 'h': t_h
                })

    # 重複を除外してTop-5を取得
    top_picks = non_max_suppression(candidates, min_dist=30, top_k=5)

    if not top_picks:
        print("候補が見つかりませんでした。ROIを変えて再試行してください。")
        sys.exit(1)

    print(f"\n★ {len(top_picks)}個の有力候補が見つかりました！")
    print("地図上に表示された候補（Rank 1〜5）を確認してください。")
    print("赤枠(Rank 1)が最有力ですが、誤差がある場合は橙・黄枠を見てください。")

    # --- 結果の可視化 ---
    if not args.no_show:
        res_img = img_ref.copy()
        
        # ランクごとの色 (BGR): 赤, 橙, 黄, 緑, 青
        colors = [
            (0, 0, 255),    # Rank 1
            (0, 165, 255),  # Rank 2
            (0, 255, 255),  # Rank 3
            (0, 255, 0),    # Rank 4
            (255, 0, 0)     # Rank 5
        ]

        print("\n--- 検出結果リスト ---")
        for i, pick in enumerate(top_picks):
            top_left = pick['pt']
            w_match = pick['w']
            h_match = pick['h']
            center_x = top_left[0] + w_match // 2
            center_y = top_left[1] + h_match // 2
            
            # 緯度経度
            lat = args.tl_lat + (center_y / ref_h) * (args.br_lat - args.tl_lat)
            lon = args.tl_lon + (center_x / ref_w) * (args.br_lon - args.tl_lon)

            print(f"Rank {i+1}: Score={pick['score']:.3f}, Lat={lat:.7f}, Lon={lon:.7f}")
            
            # 描画
            color = colors[i] if i < len(colors) else (255, 255, 255)
            cv2.rectangle(res_img, top_left, (top_left[0]+w_match, top_left[1]+h_match), color, 2)
            
            # ラベル (Rank番号)
            label = f"#{i+1}"
            cv2.putText(res_img, label, (top_left[0], top_left[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        plt.figure(figsize=(12, 8))
        plt.title("Search Results (Red=#1, Orange=#2, Yellow=#3...)")
        plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()