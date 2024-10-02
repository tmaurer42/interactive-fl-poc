export const fetchAsArrayBuffer = async (url: string) => {
	const response = await fetch(url);
	return await response.arrayBuffer();
};

export const fetchAsUint8Array = async (url: string) => {
	const arrayBuffer = await fetchAsArrayBuffer(url);
	return new Uint8Array(arrayBuffer);
};

/**
 * Shuffles an array in place using the Fisher-Yates (Knuth) algorithm.
 *
 * @param {Array} array - The array to shuffle. The original array is modified.
 * @returns {Array} The shuffled array (same reference as the input array).
 */
export function shuffleArray<T>(array: T[]) {
	for (let i = array.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[array[i], array[j]] = [array[j], array[i]];
	}

	return array;
}

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

export function getFirstMatchingProperty(obj: any, prefix: string): any {
	const keys = Object.keys(obj);
	const matchingKey = keys.find((key) => key.startsWith(prefix));

	return matchingKey ? obj[matchingKey] : undefined;
}

export function stratifiedSplit<T>(
	data: T[],
	labels: (string | number)[],
	valSize: number = 0.2
): [T[], T[]] {
	if (data.length !== labels.length) {
		throw new Error("Data and labels must have the same length");
	}

	const labelDataMap: { [key: string]: T[] } = {};

	labels.forEach((label, index) => {
		if (!labelDataMap[label]) {
			labelDataMap[label] = [];
		}
		labelDataMap[label].push(data[index]);
	});

	const trainData: T[] = [];
	const valData: T[] = [];

	for (const label in labelDataMap) {
		const allDataForLabel = labelDataMap[label];
		const totalForLabel = allDataForLabel.length;
		const valCount = Math.floor(totalForLabel * valSize);

		const shuffledData = shuffleArray(allDataForLabel);

		valData.push(...shuffledData.slice(0, valCount));
		trainData.push(...shuffledData.slice(valCount));
	}

	return [trainData, valData];
}
