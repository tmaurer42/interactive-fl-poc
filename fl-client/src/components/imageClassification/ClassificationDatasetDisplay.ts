import { VisionDatasetDisplayBase } from "../base";
import { ModelImage, Stage } from "modules";
import { ClassificationResult } from "./ClassificationResult";

export class ClassificationDatasetDisplay extends VisionDatasetDisplayBase<ClassificationResult> {
	private headerText = "Images already used for training";
	protected stage = Stage.Trained;

	protected renderHeader(): string {
		return `
			<div class="block is-flex is-flex-wrap-nowrap is-justify-content-space-between">
				<div>
					<div class="mt-2">${this.headerText}</div>
				</div>
			</div>
		`;
	}

	protected renderImageCell(
		container: HTMLDivElement,
		imageId: number,
		modelImage: ModelImage<ClassificationResult>,
		onImageLoaded: (imgElement: HTMLImageElement) => void
	): void {
		const label = modelImage.predictionResult?.label ?? "";
		const imgElement = document.createElement("img");
		imgElement.style.maxHeight = "100%";
		imgElement.id = `img-${imageId}`;
		imgElement.onload = () => onImageLoaded?.(imgElement);
		imgElement.src = modelImage.imageData;

		const imageHeight = 10;
		const imageCard = `
            <div class="card" id="image-card-${imageId}">
				<div class="card-content">
    				<div 
						class="content is-flex is-justify-content-center is-align-items-center"
						style="height:${imageHeight}em;"
					>
						${imgElement.outerHTML}
					</div>
				</div>
                <footer class="card-footer">
					<label class="card-footer-item">
						<strong>
							${label}
						</strong>
					</label>
                </footer>
            </div>
        `;

		container.innerHTML = imageCard;
	}
}
