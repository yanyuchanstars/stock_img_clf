# stock_img_clf
CNN、随机森林应用于股票四种形态A形（先涨后跌），U形（先跌后涨），R形（上涨），D形（下跌）的识别，基于keras，后端为tensorflow。
cnn_hyper.py 为超参数自动优化；someplots.py为cnn层输出和通道输出可视化，热力图可视化等；vectorization.py为一些数据预处理；random_forest.py为cnn特征提取，拟合随机森林分类器，自动调参。
