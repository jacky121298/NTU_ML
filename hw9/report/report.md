<center><font size="6" face="黑體">Machine Learning HW9</font></center>
<center><font size="3"><pre>學號:B06902060  系級:資工三  姓名:鄒宗霖</pre></font></center>

---

1. ( 3% ) 請至少使用兩種方法 ( autoencoder 架構、optimizer、data preprocessing、後續降維方法、clustering 算法等等 ) 來改進 baseline code 的 accuracy。

   a. 分別記錄改進前、後的 test accuracy 為多少。

   |               | before improvement | after improvement |
   | :-----------: | :----------------: | :---------------: |
   | test accuracy |      0.74776       |      0.77858      |

   b. 分別使用改進前、後的方法，將 val data 的降維結果 ( embedding ) 與他們對應的 label 畫出來。

   ![baseline](C:\Users\user\Desktop\Programming\3 Junior\ML\hw9-jacky12123\report\prob1\baseline.png)

   <img src="C:\Users\user\Desktop\Programming\3 Junior\ML\hw9-jacky12123\report\prob1\embedding.png" alt="embedding" style="zoom:72%;" />

   c. 盡量詳細說明你做了哪些改進。

   我使用了下列兩種方法提高 testing data 的正確率，首先是改善 autoencoder 的架構，我將 self.encoder 的 convolution 層數從三層變成六層，如下面的程式碼所示；接著是改善降維的方法，我先利用 KernelPCA 把 latents 從 4096 維降到 500 維，接著用 PCA 從 500 維降到 64 維再到 16 維，最後再用 TSNE 降到 2 維，才不會一次把維度降的太多，使得資料在降維的過程中遺漏的太多。

   ```python
   nn.Conv2d(3, 64, 3, stride = 1, padding = 1),
   nn.ReLU(True),
   
   nn.Conv2d(64, 64, 3, stride = 1, padding = 1),
   nn.ReLU(True),
   nn.MaxPool2d(2),
   
   nn.Conv2d(64, 128, 3, stride = 1, padding = 1),
   nn.ReLU(True),
   
   nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
   nn.ReLU(True),
   nn.MaxPool2d(2),
   
   nn.Conv2d(128, 256, 3, stride = 1, padding = 1),
   nn.ReLU(True),
   
   nn.Conv2d(256, 256, 3, stride = 1, padding = 1),
   nn.ReLU(True),
   nn.MaxPool2d(2)
   ```

2. ( 1% ) 使用你 test accuracy 最高的 autoencoder，從 trainX 中，取出 index 1, 2, 3, 6, 7, 9 這 6 張圖片，畫出他們的原圖以及 reconstruct 之後的圖片。

   <img src="C:\Users\user\Desktop\Programming\3 Junior\ML\hw9-jacky12123\report\prob2\reconstruct.png" alt="reconstruct" style="zoom:60%;" />

3. ( 2% ) 在 autoencoder 的訓練過程中，至少挑選 10 個 checkpoints

   a. 請用 model 的 train reconstruction error ( 用所有的 trainX 計算 MSE ) 和 val accuracy 對那些 checkpoints 作圖。

   <img src="C:\Users\user\Desktop\Programming\3 Junior\ML\hw9-jacky12123\report\prob3\checkpoints.png" alt="checkpoints" style="zoom:70%;" />
   
   b. 簡單說明你觀察到的現象。
   
   隨著 # of epoch 的增加，model 的 train reconstruction error 逐漸降低，但是 val accuracy 並沒有穩定的上升，而是呈鋸齒狀上升，可能的原因是因為 reconstruction error 的高低與 encoder , decoder 都有相關，encoder , decoder 學得越好，reconstruction error 越低；然而 val accuracy 只和 encoder 產生的 latents 有關，latents 越具有代表性，分類的結果越明確、val accuracy 越高。所以有可能在 reconstruction error 降低的某個過程中，是因為 decoder 學得越來越好，然而 encoder 所產生的 latents 並不具代表性，導致 val  accuracy 不增反降。

