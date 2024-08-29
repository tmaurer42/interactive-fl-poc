export const fetchAsArrayBuffer = async (url: string) => {
	const response = await fetch(url);
	return await response.arrayBuffer();
};

export const fetchAsUint8Array = async (url: string) => {
	const arrayBuffer = await fetchAsArrayBuffer(url);
	return new Uint8Array(arrayBuffer);
};
