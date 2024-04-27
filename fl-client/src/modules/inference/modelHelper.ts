export type ClassProbability = {
	label: string;
	probability: number;
};

/**
 * Get the top K classes from the output data.
 * @param outputData
 * The output data from the model.
 * @param topK
 * The number of top classes to get.
 * @param classes
 * The classes of the model.
 * @returns
 * The top K classes.
 */
const classesTopK = (
	outputData: number[],
	topK: number,
	classes: string[]
): ClassProbability[] => {
	//Create an array of indices [0, 1, 2, ..., 999].
	var resultArray = Array.from(Array(outputData.length).keys());
	//Sort the indices based on the output data.
	resultArray.sort((a, b) => outputData[b] - outputData[a]);
	//Get the top K indices.
	return resultArray.slice(0, topK).map((i) => {
		return { label: classes[i], probability: outputData[i] };
	});
};

/**
 * Softmax function
 *
 * The softmax function is used to normalize the output of a network
 * to a probability distribution over predicted output classes.
 * @param resultArray
 * The resultArray is the output of the model.
 * @returns
 * The softmax of the resultArray.
 */
const softmax = (resultArray: number[]) => {
	// Get the largest value in the array.
	const largestNumber = Math.max(...resultArray);
	// Apply exponential function to each result item subtracted by the largest number, use reduce to get the previous result number and the current number to sum all the exponentials results.
	const sumOfExp = resultArray
		.map((resultItem) => Math.exp(resultItem - largestNumber))
		.reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
	//Normalizes the resultArray by dividing by the sum of all exponentials; this normalization ensures that the sum of the components of the output vector is 1.
	return resultArray.map((resultValue) => {
		return Math.exp(resultValue - largestNumber) / sumOfExp;
	});
};

export default { classesTopK, softmax };
