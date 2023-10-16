<center><font size="6" face="黑體">Machine Learning HW12</font></center>
<center><font size="3"><pre>學號:B06902060  系級:資工三  姓名:鄒宗霖</pre></font></center>

---

1. 請描述你實作的模型架構、方法以及 accuracy 為何。其中你的方法必須為 domain adversarial training 系列。(就是你的方法必須要讓輸入 training data & testing data 後的某一層輸出 domain 要相近) (2%)

   模型架構分成三個部分，分別為 Feature Extractor, Label Predictor, Domain Classifier，其中 Feature Extractor 為 VGG-like 疊法，channel 數依序變成 1, 64, 128, 256, 256, 512, 512；Label Predictor 為 Multilayer Perceptron，有兩層 nn.Linear(512, 512) 以及一層 nn.Linear(512, 10)；Domain Classifier 為 Multilayer Perceptron，有五層 nn.Linear(512, 512), nn.BatchNorm1d(512)以及一層 nn.Linear(512, 1)。採用的方法為使用原始論文中 adaptive parameter lambda 訓練 2000 epochs，在每一個 epoch 裡面又分成兩個階段，第一個階段是訓練 Domain Classifier；第二個階段是訓練 Feature Extractor, Label Predictor，其中所使用的 loss function 為 $class\ cross\ entropy - \lambda * domain\ binary\ cross\ entropy $。Kaggle 上的正確率為 0.75394。

2. 請視覺化真實圖片以及手繪圖片通過沒有使用 domain adversarial training 的 feature extractor 的 domain 分布圖。(2%)

   <img src="C:\Users\user\Desktop\Programming\3 Junior\ML\hw12-jacky12123\report\prob2\prob2.png" alt="prob2" style="zoom:72%;" />

   在上圖中，我們可以看得 source data 被分成明顯的十群，然而 target data 並沒有明顯分群的現象，可能的原因是此題的 Feature Extractor 看過很多筆 source data，所以抽出來的 feature 就頗具意義；但是此題的 Feature Extractor 沒看過 target data，導致抽出來的 feature 沒什麼意義。

3. 請視覺化真實圖片以及手繪圖片通過有使用 domain adversarial training 的 feature extractor 的 domain 分布圖。(2%)

   <img src="C:\Users\user\Desktop\Programming\3 Junior\ML\hw12-jacky12123\report\prob3\prob3.png" alt="prob3" style="zoom:72%;" />
   
   在上圖中，我們可以看得 source data 和 target data 都被分成明顯的十群，可能的原因是在此題的 model 中我們多加了 Domain Classifier，好讓 Feature Extractor 學習如何產生 feature 騙過 Domain Classifier，長久下來，不管是 source data 或 target data ，都會被 Feature Extractor 轉換到同一個 domain 中。

 