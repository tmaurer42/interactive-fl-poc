export type KeyValuePairs = { [key: string]: any };

export enum Stage {
	Inference,
	ReadyForTraining,
	Trained,
	Testing,
}

export interface ModelImage<T extends KeyValuePairs> {
	imageData: string;
	stage: Stage;
	metadata: KeyValuePairs;
	predictionResult?: T;
}

export type ModelImageCreateInput<T extends KeyValuePairs> = Omit<
	ModelImage<T>,
	"id"
>;
export type ModelImageUpdateInput<T extends KeyValuePairs> = Partial<
	Omit<ModelImage<T>, "imageData">
>;

export class ImageRepository {
	private db: IDBDatabase | null = null;
	private _objectStoreName: string = "";

	get objectStoreName() {
		return this._objectStoreName;
	}

	constructor() {}

	/**
	 * Initialize the database with the given object store name.
	 * @param objectStoreName Name of the object store to save the data to.
	 * @returns A promise that resolves when the db has been initialized.
	 */
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
				}).createIndex("stage", "stage");
			};

			request.onsuccess = (event: Event) => {
				this.db = (event.target as IDBRequest).result as IDBDatabase;
				resolve();
			};
		});
	}

	/**
	 * Add a ModelImage to the db.
	 * @param input The ModelImage to store.
	 * @returns A promise that resolves with the id of the new ModelImage.
	 */
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

	/**
	 * Gets a ModelImage from the db.
	 * @param id The id of the ModelImage.
	 * @returns A promise that resolves with the ModelImages data.
	 */
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

	/**
	 * Gets ModelImages from a list of ids.
	 * @param ids The ids of the ModelImages to retrieve from the db.
	 * @returns A promise that resolves with an array of ModelImages.
	 */
	public getImagesByIds<T extends KeyValuePairs>(
		ids: number[]
	): Promise<(ModelImage<T> | undefined)[]> {
		return new Promise((resolve, reject) => {
			if (!this.db) {
				return reject("Database not initialized");
			}

			const transaction = this.db.transaction(
				this.objectStoreName,
				"readonly"
			);
			const objectStore = transaction.objectStore(this.objectStoreName);

			const promises = ids.map(
				(id) =>
					new Promise<ModelImage<T> | undefined>(
						(resolve, reject) => {
							const request = objectStore.get(id);

							request.onsuccess = () => {
								resolve(request.result as ModelImage<T>);
							};

							request.onerror = (event: Event) => {
								reject((event.target as IDBRequest).error);
							};
						}
					)
			);

			Promise.all(promises)
				.then((results) => resolve(results))
				.catch((error) => reject(error));
		});
	}

	/**
	 * Gets all ModelImage ids in the db. If stage is provided, only gets those
	 * in that specific stage.
	 * @param stage (optional) The Stage to filter for.
	 * @returns A promise that resolves with the retrieved ids.
	 */
	getAllIds(stage?: Stage): Promise<number[]> {
		return new Promise((resolve, reject) => {
			if (!this.db) {
				return reject("Database not initialized");
			}

			const transaction = this.db.transaction(
				this.objectStoreName,
				"readonly"
			);
			const objectStore = transaction.objectStore(this.objectStoreName);

			let request = undefined;
			if (stage !== undefined) {
				const index = objectStore.index("stage");
				const query = IDBKeyRange.only(stage);
				request = index.getAllKeys(query);
			} else {
				request = objectStore.getAllKeys();
			}

			request.onsuccess = () => {
				resolve(request.result as number[]);
			};

			request.onerror = (event: Event) => {
				reject((event.target as IDBRequest).error);
			};
		});
	}

	/**
	 * Updates a ModelImage. Any props not provided in the input
	 * will be unchanged in the existing ModelImage.
	 * @param id Id of the ModelImage to update.
	 * @param input Update for the ModelImage
	 * @returns A promise that resolves when the ModelImage was updated.
	 */
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
					image.metadata = input.metadata ?? image.metadata;
					image.predictionResult =
						input.predictionResult ?? image.predictionResult;
					image.stage = input.stage ?? image.stage;
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

	/**
	 * Deletes a ModelImage from the db.
	 * @param id The id of the ModelImage to be deleted.
	 * @returns A promise that resolves when the ModelImage is deleted.
	 */
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

	/**
	 * Gets all ModelImages in the db. If stage is provided, only gets those
	 * in that specific stage.
	 * @param stage (optional) The Stage to filter for.
	 * @returns A promise that resolves with the retrieved ModelImages.
	 */
	getAllImages<T extends KeyValuePairs>(
		stage?: Stage
	): Promise<ModelImage<T>[]> {
		return new Promise((resolve, reject) => {
			if (!this.db) {
				return reject("Database not initialized");
			}

			const transaction = this.db.transaction(
				this.objectStoreName,
				"readonly"
			);
			const objectStore = transaction.objectStore(this.objectStoreName);
			let request = undefined;
			if (stage !== undefined) {
				const index = objectStore.index("stage");
				const query = IDBKeyRange.only(stage);
				request = index.getAll(query);
			} else {
				request = objectStore.getAll();
			}

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
