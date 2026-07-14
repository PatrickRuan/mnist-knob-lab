# 神經網路旋鈕實驗室（mnist-knob-lab）

一個互動式的深度神經網路（DNN）教學網頁，讓學生「親手轉旋鈕」體會**梯度下降**的直覺。

網路結構是 **64 → 10 → 10 → 2**：輸入一張 8×8 的手寫數字圖片（0 或 1），輸出二分類。
訓練好之後，把最後一層 10×2 的 20 個權重打亂成隨機值、前三層凍結，讓學生透過 20 個旋鈕
手動調整——誤差變大就往反方向轉，這正是電腦訓練神經網路時做的事。

## 直接開始

整個教具是**單一自包含的 [`index.html`](index.html)**，權重與測試資料都已嵌入其中，
無任何外部相依。要用只需要這一個檔案：

- 下載 `index.html` 後雙擊開啟（離線也能用），或
- 直接開線上版：**https://patrickruan.github.io/mnist-knob-lab/**

## 六個分頁

1. **網路如何判斷** — 用真實測試圖片示範前向傳播：資料流過經典的節點連線圖，
   輸出層兩個節點各給出機率，機率大的勝出。另附手寫畫板（自己畫 0/1 即時判定）、
   第一層權重熱圖（看神經元在偵測什麼特徵），以及訓練用的 Python 原始碼。
2. **旋鈕挑戰** — 20 個旋鈕對應最後一層權重，手動調整觀察誤差即時變化；
   有「讓電腦調一步」按鈕用真實梯度自動更新，對比人手與演算法的差異。
3. **這頁怎麼做的** — 整個教具用 Claude Code 以自然語言對話產生的過程記錄（vibe coding 教學）。
4. **GPT 怎麼學** — 自監督式學習：一句話自動展開成好幾筆「文字接龍」訓練資料，
   並用內嵌語料現場訓練一個可互動的 bigram 語言模型。
5. **詞的地圖與注意力** — word embedding（全語料統計）vs attention（當下句子）
   的對照實驗，含可點擊的 2D 詞地圖與「蘋果」多義詞的注意力弧線視覺化。
6. **訓練詞向量** — word2vec / skip-gram：訓練一層 hidden layer 的網路玩「猜鄰居」，
   取輸入層權重當詞向量，在瀏覽器裡現場跑梯度下降，即時看地圖從亂到聚。

## 檔案說明

| 檔案 | 用途 |
|------|------|
| `index.html` | 教具本體，單一自包含檔案，**交付給學生只需要這個** |
| `train.py` | 預訓練腳本（numpy 手刻 backprop），產出 `weights.json` |
| `weights.json` | `train.py` 的輸出（權重＋測試資料），建置時已嵌入 `index.html` |
| `word2vec.py` | word2vec / skip-gram 教學腳本（tab ⑥ 用同一份語料），可獨立執行 |

## 重新訓練

若要換分類任務（例如 3 vs 8）或調整網路：

```bash
pip install numpy scikit-learn
python train.py        # 產出 weights.json
```

再把 `weights.json` 的內容注入 `index.html` 中的 `DATA` 常數即可。

想單獨玩 word2vec，可直接執行：

```bash
python word2vec.py     # 印出訓練過程、相似度與最近鄰
```

## 技術

純 vanilla HTML / CSS / JavaScript，SVG 與 Canvas 繪圖，零外部相依。
資料集為 scikit-learn 的 `load_digits`（本身即 8×8），篩選 0 與 1 兩類。
