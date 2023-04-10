# db-compression-ml

## Steps
1. Exporting the bit-perfect database from C in a format of maybe CSV
2. Selecting Features: Identify the features in your bit-perfect database that will be used to train the machine learning model. 
3. Choosing a Machine Learning Algorithm: Select a machine learning algorithm that is suitable for the type of problem you are trying to solve. Thinking about using neural network algorithm
4. Training the Model: Use the selected algorithm to train your machine learning model on the prepared bit-perfect database that got converted to CSV. 
5. Evaluating the Model: After training, evaluate the performance of the model using a test set. This will help you determine how well the model is performing and identify any areas where it can be improved.
6. Tuning the Model: If necessary, adjust the parameters of the machine learning algorithm and retrain the model to improve its performance.
7. Deploying the Model: Once the model has been trained and tested, deploy it into production to make predictions on new data.

<br />

## [What is bit-perfect database](https://nyc.cs.berkeley.edu/wiki/Bit-Perfect_DB)

#Example: <br/>
A slice with the following slots<br/>
And with the respective sizes in bits<br/>
| VALUE | VISITED | MEX | REMOTENESS |
|-------|-------|-------|-------|
| 2 | 1 | 5  |     8      |

#Steps to convert bit-perfect database from dat.gz to CSV <br />

Here, the BlankOXToPosiiton function is a unhash function which is value in the dpdb
```c
POSITION BlankOXToPosition(theBlankOX)
BlankOX *theBlankOX;
{
	int i;
	POSITION position = 0;

	for(i = 0; i < BOARDSIZE; i++)
		position += g3Array[i] * (int)theBlankOX[i]; /* was (int)position... */

	return(position);
}
```
