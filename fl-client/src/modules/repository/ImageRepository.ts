type KeyValuePairs = { [key: string]: any };

export enum Stage {
	Inference,
	ReadyForTraining,
}

export interface ModelImage<T extends KeyValuePairs> {
	imageData: string; // Base64 encoded image data
	stage: Stage;
	metadata: KeyValuePairs; // Metadata associated with the image
	predictionResult?: T;
}

export type ModelImageCreateInput<T extends KeyValuePairs> = Omit<
	ModelImage<T>,
	"id"
>;
export type ModelImageUpdateInput<T extends KeyValuePairs> = Omit<
	ModelImage<T>,
	"imageData"
>;

export class ImageRepository {
	private db: IDBDatabase | null = null;
	private _objectStoreName: string = "";

	get objectStoreName() {
		return this._objectStoreName;
	}

	constructor() {}

	public initializeDB(objectStoreName: string): Promise<void> {
		this._objectStoreName = objectStoreName;
		return new Promise((resolve, reject) => {
			const request = indexedDB.open("imageUploaderDB", 1);

			request.onerror = (event: Event) => {
				console.error(
					"Database error: ",
					(event.target as IDBRequest).error
				);
				reject((event.target as IDBRequest).error);
			};

			request.onupgradeneeded = (event: IDBVersionChangeEvent) => {
				const db = (event.target as IDBRequest).result as IDBDatabase;
				db.createObjectStore(this.objectStoreName, {
					autoIncrement: true,
				});
			};

			request.onsuccess = (event: Event) => {
				this.db = (event.target as IDBRequest).result as IDBDatabase;
				resolve();
			};
		});
	}

	addImage<T extends KeyValuePairs>(
		input: ModelImageCreateInput<T>
	): Promise<number> {
		return new Promise((resolve, reject) => {
			if (!this.db) {
				return reject("Database not initialized");
			}

			const transaction = this.db.transaction(
				this.objectStoreName,
				"readwrite"
			);
			const objectStore = transaction.objectStore(this.objectStoreName);
			const request = objectStore.add(input);

			request.onsuccess = () => {
				resolve(request.result as number);
			};

			request.onerror = (event: Event) => {
				reject((event.target as IDBRequest).error);
			};
		});
	}

	getImage<T extends KeyValuePairs>(
		id: number
	): Promise<ModelImage<T> | undefined> {
		return new Promise((resolve, reject) => {
			if (!this.db) {
				return reject("Database not initialized");
			}

			const transaction = this.db.transaction(
				this.objectStoreName,
				"readonly"
			);
			const objectStore = transaction.objectStore(this.objectStoreName);
			const request = objectStore.get(id);

			request.onsuccess = () => {
				resolve(request.result as ModelImage<T>);
			};

			request.onerror = (event: Event) => {
				reject((event.target as IDBRequest).error);
			};
		});
	}

	getAllIds(): Promise<number[]> {
		return new Promise((resolve, reject) => {
			if (!this.db) {
				return reject("Database not initialized");
			}

			const transaction = this.db.transaction(
				this.objectStoreName,
				"readonly"
			);
			const objectStore = transaction.objectStore(this.objectStoreName);
			const request = objectStore.getAllKeys();

			request.onsuccess = () => {
				resolve(request.result as number[]);
			};

			request.onerror = (event: Event) => {
				reject((event.target as IDBRequest).error);
			};
		});
	}

	updateImageData<T extends KeyValuePairs>(
		id: number,
		input: ModelImageUpdateInput<T>
	): Promise<void> {
		return new Promise((resolve, reject) => {
			if (!this.db) {
				return reject("Database not initialized");
			}

			const transaction = this.db.transaction(
				this.objectStoreName,
				"readwrite"
			);
			const objectStore = transaction.objectStore(this.objectStoreName);
			const request = objectStore.get(id);

			request.onsuccess = () => {
				const image = request.result as ModelImage<any>;
				if (image) {
					image.metadata = input.metadata;
					image.predictionResult = input.predictionResult;
					const updateRequest = objectStore.put(image, id);

					updateRequest.onsuccess = () => {
						resolve();
					};

					updateRequest.onerror = (event: Event) => {
						reject((event.target as IDBRequest).error);
					};
				} else {
					reject("Image not found");
				}
			};

			request.onerror = (event: Event) => {
				reject((event.target as IDBRequest).error);
			};
		});
	}

	deleteImage(id: number): Promise<void> {
		return new Promise((resolve, reject) => {
			if (!this.db) {
				return reject("Database not initialized");
			}

			const transaction = this.db.transaction(
				this.objectStoreName,
				"readwrite"
			);
			const objectStore = transaction.objectStore(this.objectStoreName);
			const request = objectStore.delete(id);

			request.onsuccess = () => {
				resolve();
			};

			request.onerror = (event: Event) => {
				reject((event.target as IDBRequest).error);
			};
		});
	}

	getAllImages<T extends KeyValuePairs>(): Promise<ModelImage<T>[]> {
		return new Promise((resolve, reject) => {
			if (!this.db) {
				return reject("Database not initialized");
			}

			const transaction = this.db.transaction(
				this.objectStoreName,
				"readonly"
			);
			const objectStore = transaction.objectStore(this.objectStoreName);
			const request = objectStore.getAll();

			request.onsuccess = () => {
				resolve(request.result as ModelImage<T>[]);
			};

			request.onerror = (event: Event) => {
				reject((event.target as IDBRequest).error);
			};
		});
	}
}

export const Repository = new ImageRepository();
