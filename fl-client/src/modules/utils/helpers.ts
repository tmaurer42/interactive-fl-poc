/**
 * Performs a GET request on the given url and returns the result
 * as an ArrayBuffer
 * @param url The url tp fetch from
 * @returns Response content as ArrayBuffer
 */
export const fetchAsArrayBuffer = async (url: string) => {
	const response = await fetch(url);
	return await response.arrayBuffer();
};

/**
 * Performs a GET request on the given url and returns the result
 * as a Uint8Array
 * @param url The url tp fetch from
 * @returns Response content as Uint8Array
 */
export const fetchAsUint8Array = async (url: string) => {
	const arrayBuffer = await fetchAsArrayBuffer(url);
	return new Uint8Array(arrayBuffer);
};

/**
 * Shuffles an array in place using the Fisher-Yates (Knuth) algorithm.
 * For convenience, the input array is also returned.
 * @param array The array to be shuffled
 * @returns Shuffled input array
 */
export function shuffleArray<T>(array: T[]) {
	for (let i = array.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[array[i], array[j]] = [array[j], array[i]];
	}

	return array;
}

/**
 * Yields batches of the given size. from an array.
 * @param array The array to batchify.
 * @param batchSize Size of the batches.
 * @returns A generator to iterate over the batches.
 */
export function* batchify<T>(array: T[], batchSize: number) {
	const numBatches = Math.ceil(array.length / batchSize);

	for (let i = 0; i < numBatches; i++) {
		yield array.slice(i * batchSize, i * batchSize + batchSize);
	}
}

/**
 * Check if an Element has all provided attributes.
 * An error is logged to the console for each missing attribute.
 * @param element The element to check for attributes
 * @param attributes List of attribute names tp check
 * @returns Whether or not all attributes are present on the given Element
 */
export const hasRequiredAttributes = (
	element: HTMLElement,
	attributes: string[]
) => {
	let result = true;

	for (const attribute of attributes) {
		if (!element.hasAttribute(attribute)) {
			console.error(
				`missing attribute in ${element.localName}: ${attribute}`
			);
			result = false;
		}
	}

	return result;
};

/**
 * Gets the first property from an object that start with a specific string.
 * @param obj The object from which to retrieve the property
 * @param prefix Start of the property name
 * @returns Property value or undefined, if no property name starts with obj
 */
export function getFirstMatchingProperty(obj: any, prefix: string): any {
	const keys = Object.keys(obj);
	const matchingKey = keys.find((key) => key.startsWith(prefix));

	return matchingKey ? obj[matchingKey] : undefined;
}

/**
 * Reshapes a 1D array into an n-dimensional nested array based on the given dimensions.
 * @param dimensions
 * @param array
 * @returns The reshaped array.
 */
export function reshapeArray(dimensions: number[], array: number[]): any[] {
	if (dimensions.length === 0) return array;

	const totalElements = dimensions.reduce((a, b) => a * b, 1);
	if (array.length !== totalElements) {
		throw new Error(
			"The total elements in the array do not match the given dimensions"
		);
	}

	function createNestedArray(dimensions: number[], array: number[]): any[] {
		if (dimensions.length === 1) {
			return array.slice(0, dimensions[0]);
		}

		const size = dimensions[0];
		const subArraySize = array.length / size;

		const result = [];
		for (let i = 0; i < size; i++) {
			result.push(
				createNestedArray(
					dimensions.slice(1),
					array.slice(i * subArraySize, (i + 1) * subArraySize)
				)
			);
		}

		return result;
	}

	return createNestedArray(dimensions, array);
}
