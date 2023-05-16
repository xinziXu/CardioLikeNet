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
Or you can from scrach as in read_
