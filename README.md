# nodebase
***Dependent libraries***

numpy>=1.21.5

pandas>=1.4.4

scikit-learn>=1.0.2


***Operation method***

First, go to the code folder, after which run the following code in the terminal.

```shell
python nodebase.py -n cora -c 0.059 -e 1
```

***Parameter explanation***

"-n": 

Selected dataset. The default is cora.

"-c":

It is hyperparameter. The best hyperparameter for Cora is 0.059, for Cite is 0.024, and for Pub is 0.062.

The default is 0.059.

"-e":

This parameter is the number of times the algorithm performance will be evaluated. The default is 1 time.
