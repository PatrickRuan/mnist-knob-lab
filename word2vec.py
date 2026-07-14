"""迷你 word2vec（skip-gram）：單一 hidden layer 的網路，取輸入層權重當詞向量。

用 66 句預先斷詞的小語料訓練，視窗 ±2 產生（中心詞 → 鄰居）訓練對，
全 softmax cross-entropy + SGD。編碼維度 D=4，訓練完的 W1 每一列
就是一個詞的 4 維詞向量（D=2 時 102 個詞塞進 2 維會塌縮、失去區分度，
D=4 是實測後品質與訓練速度的甜蜜點）；要畫成散佈圖時再用 PCA 投影成 2D。
"""
import numpy as np

CORPUS = """南韓 男星 李鐘碩 與 歌手 IU李知恩 爆出 熱戀
南韓 歌手 IU李知恩 推出 新專輯
男星 李鐘碩 主演 新戲
歌手 IU李知恩 獲得 大獎
南韓 男星 主演 電影 爆出 好評
我們 老師 今天 教 神經網路
神經網路 會 預測 下一個 詞
我們 今天 上課 學 文字接龍
老師 說 模型 讀 愈多 資料 愈 聰明
模型 會 預測 下一個 詞
台灣 棒球隊 贏得 世界 冠軍
台灣 珍珠奶茶 風靡 世界
南韓 女星 與 男星 爆出 緋聞
南韓 天團 來 台灣 開 演唱會
南韓 電影 獲得 國際 大獎
男星 與 女星 爆出 婚訊
女星 主演 新戲 獲得 好評
歌手 推出 新專輯 獲得 好評
天團 推出 新專輯 風靡 世界
李鐘碩 與 IU李知恩 合作 新戲
IU李知恩 主演 電影 獲得 大獎
新戲 獲得 觀眾 好評
電影 獲得 觀眾 好評
觀眾 說 新戲 很 好看
記者 爆出 男星 與 歌手 熱戀
記者 說 南韓 天團 要 來 台灣
台灣 觀眾 喜歡 南韓 電影
台灣 歌手 推出 新專輯
世界 冠軍 是 台灣 棒球隊
棒球隊 今天 贏得 比賽
我們 今天 看 棒球 比賽
我們 老師 說 神經網路 很 有趣
老師 今天 教 我們 梯度下降
梯度下降 會 降低 誤差
神經網路 靠 梯度下降 學習
模型 靠 資料 學習
模型 讀 愈多 資料 愈 聰明
資料 愈多 模型 愈 聰明
我們 上課 學 神經網路
學生 上課 學 文字接龍
學生 說 文字接龍 很 有趣
文字接龍 就是 預測 下一個 詞
預測 下一個 詞 就是 自監督 學習
自監督 學習 不用 人工 標註
人工 標註 很 花 時間
GPT 靠 文字接龍 學習
GPT 讀 愈多 資料 愈 聰明
GPT 會 預測 下一個 詞
下一個 詞 由 機率 決定
機率 大 的 詞 常 被 選中
老師 說 GPT 就是 文字接龍
我們 喜歡 上 老師 的 課
珍珠奶茶 是 台灣 的 驕傲
世界 都 喜歡 台灣 珍珠奶茶
演唱會 門票 很 快 賣完
天團 演唱會 門票 賣完
歌手 在 演唱會 演唱 新歌
新歌 獲得 歌迷 好評
歌迷 喜歡 IU李知恩 的 新歌
熱戀 消息 獲得 歌迷 祝福
蘋果 很 好吃
蘋果 是 好吃 的 水果
我們 喜歡 吃 蘋果
蘋果 推出 新手機
蘋果 新手機 賣完
新手機 獲得 好評"""

WIN, D, LR, EPOCHS, SEED = 2, 4, 0.05, 60, 1

sents = [line.split() for line in CORPUS.strip().split("\n")]
vocab = sorted({w for s in sents for w in s})
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# skip-gram 訓練對：(中心詞, 視窗內的鄰居)
pairs = np.array([
    (idx[s[i]], idx[s[j]])
    for s in sents
    for i in range(len(s))
    for j in range(max(0, i - WIN), min(len(s), i + WIN + 1))
    if j != i
])
print(f"詞彙量 V={V}，訓練對 {len(pairs)} 筆")

rng = np.random.default_rng(SEED)
W1 = rng.normal(0, 0.5, (V, D))   # 輸入層權重：訓練完就是詞向量！
W2 = rng.normal(0, 0.5, (D, V))   # 輸出層權重：訓練完就丟掉

for epoch in range(EPOCHS):
    rng.shuffle(pairs)
    loss = 0.0
    for c, o in pairs:                    # c=中心詞、o=要猜的鄰居
        h = W1[c]                         # one-hot 選中的那一列（前向傳播）
        z = h @ W2
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()                      # softmax：對每個詞的預測機率
        loss += -np.log(p[o] + 1e-12)     # cross-entropy
        p[o] -= 1.0                       # 反向傳播（dz）
        dW1 = p @ W2.T
        W2 -= LR * np.outer(h, p)
        W1[c] -= LR * dW1                 # 只有被選中的那一列會更新
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:3d}  平均 loss = {loss / len(pairs):.4f}")

def cos(a, b):
    va, vb = W1[idx[a]], W1[idx[b]]
    return va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)

def neighbors(w, k=4):
    v = W1[idx[w]]
    sims = W1 @ v / (np.linalg.norm(W1, axis=1) * np.linalg.norm(v) + 1e-12)
    order = np.argsort(-sims)
    return [(vocab[i], round(float(sims[i]), 2)) for i in order[1:k + 1]]

print("\n=== 貓狗實驗（上下文相同的詞，權重被訓練得幾乎一樣）===")
print(f"cos(男星, 女星)     = {cos('男星', '女星'):+.2f}   ← 應接近 +1")
print(f"cos(熱戀, 緋聞)     = {cos('熱戀', '緋聞'):+.2f}   ← 應接近 +1")
print(f"cos(男星, 珍珠奶茶) = {cos('男星', '珍珠奶茶'):+.2f}   ← 不相關，應該低")

print("\n=== 最近鄰 ===")
for w in ("男星", "蘋果", "GPT", "台灣"):
    print(f"{w} → {neighbors(w)}")

print("\n=== PCA 投影成 2D（網頁版地圖用同樣的方法畫散佈圖）===")
Wc = W1 - W1.mean(axis=0)
U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
coords2d = U[:, :2] * S[:2]
for w in ("男星", "女星", "蘋果", "台灣"):
    x, y = coords2d[idx[w]]
    print(f"{w}: ({x:+.2f}, {y:+.2f})")

