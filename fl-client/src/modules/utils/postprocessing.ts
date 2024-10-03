export type ClassProbability = {
	label: string;
	probability: number;
};

/**
 * Get the top K classes from the output data.
 * @param outputData The output data from the model.
 * @param topK The number of top classes to get.
 * @param classes The classes of the model.
 * @returns The top K classes.
 */
export const classesTopK = (
	outputData: number[],
	topK: number,
	classes: string[]
): ClassProbability[] => {
	var resultArray = Array.from(Array(outputData.length).keys());
	resultArray.sort((a, b) => outputData[b] - outputData[a]);
	return resultArray.slice(0, topK).map((i) => {
		return { label: classes[i], probability: outputData[i] };
	});
};

/**
 * Softmax function
 *
 * The softmax function is used to normalize the output of a network
 * to a probability distribution over predicted output classes.
 * @param resultArray The resultArray is the output of the model.
 * @returns The softmax of the resultArray.
 */
export const softmax = (resultArray: number[]) => {
	const largestNumber = Math.max(...resultArray);
	const sumOfExp = resultArray
		.map((resultItem) => Math.exp(resultItem - largestNumber))
		.reduce((prevNumber, currentNumber) => prevNumber + currentNumber);

	return resultArray.map((resultValue) => {
		return Math.exp(resultValue - largestNumber) / sumOfExp;
	});
};
