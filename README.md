# ECG-Classification-for-summer-research
## 1. Database
### Database Information
Here is the website for the MIT-BIH database downloading: https://physionet.org/content/mitdb/1.0.0/.
And you can get more familiar with the data you will use by reading the annotation methods at https://archive.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm.<br>
### Reading Data
We can easily read the samples and their annotations by using `WFDB software package`. It support both Python and Matlab. The documentation is at
https://archive.physionet.org/physiotools/wfdb.shtml.
Below is a matlab example and you can also find the same usage in `denoising.m` and `get_anno.m`.<br>
```Matlab
id = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234];
for i = 1:48 
  data_file = strcat('mitdb/', num2str(id(i)));
  data = rdsamp(data_file, [], [], [], [3], []);
  [ann, anntype] = rdann(data_file, 'atr', []);
end
```
Or you can build it from scrach as in `read_data.m`.
## 2. Preprocessing
The pre-processing method is described in the paper `"A 2.66µW Clinician-like Cardiac Arrhythmia Watchdog Based on P-QRS-T for Wearable Applications"` including denoising, R-peak detection, and compression scheme.<br>
With the `WFDB` installed, you can run `denoising.m`, `get_anno.m`, `segmentation.m` and `features.m` sequentially to obtain the pre-processed data and annotations.<br>
Since the compression scheme was implemented directly in verilog by comparing the variation of slope defined in Equation (12) in the paper and the distances and amplitutes of the samples, you need to add this feature by yourself in the matlab code `features.m`.
## 3. ANN Classifier
You can learn the priciples of ANN in http://neuralnetworksanddeeplearning.com/chap2.html. 
You can build the model from scrach as in `MINSTtutorial.rar` with Matlab, which is helpful for you to learn the neural network knowledge more deeply. <br>
Or you can use popular deep learning frameworks like tensorflow or pytorch with python, which can help you build different types of nueral network fast. The uploaded code written by me uses pytorch. 


