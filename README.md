iScape Sensor Analysis Framework
=======

Welcome to the first version of the iScape Sensor Analysis tools. The framework is built with the purpose of helping on the calibration process of the sensors on field tests and it aim to be the primary tool for manipulating the sensors data. The data analysis tools are based on [Pandas](http://pandas.pydata.org/) and are ready to later support [Scikit](http://scikit-learn.org/stable/index.html).

![](https://i.imgur.com/CvUuWpL.gif)

The current notebook supports data from the iScape Citizen Sensors we provided you but we will add support to integrate the data from existing equipment. Currently all the data is loaded as CSV files but it is also ready to get live data directly from the Smart Citizen API in the future.

The primary goal of the tools is to help us validate the different iScape sensors and calculate their calibration values that later might automatically applied to the data the sensors push online.

How to start https://hackmd.io/s/Hkb-Cw0rb