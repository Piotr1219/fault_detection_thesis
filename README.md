# Fault detection software, based on clustering, neural networks and other machine learning methods.

The aim of the project was to implement various methods for detecting anomaly in measurements, and choose best. 

Data used in project came from car sensors. It inclused measurements with 1 second interval from various sensors. Measurands included vehicle speed, prossure in manifold, temperature of air, temperature of coolant, and some other. Collected datasets inclueded data from 9 to 11 sensors. 

![correlations](/images/correlations.png)

First, methods based on clustering samples as points in multidimensional space were examined. It was also necessary to use data dimension reduction methods. Then, several other methods were implemented, utilizing very different principles. A well-known autoregressive model was used as a kind of reference point. The algorithms used were based on e.g. LSTM neural networks, reconstruction of data from reduced space, or XGBoost architecture using decision trees. There were implemented methods that base classification only on measurements for currently analyzed moment of time, as well as those that take the sequence of 𝑛 preceding data samples as input.

The effectiveness of the methods has been checked for various types of errors, datasets, or parameters to which a disturbance has been added. This allowed the selection of methods that ensure the highest efficiency of anomaly detection, and whose operation is the most stable.

![f1_score](/images/f1_score.png)
