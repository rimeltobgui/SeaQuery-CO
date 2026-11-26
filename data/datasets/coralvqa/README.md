# SeaQuery CoralVQA evaluation subset

This folder contains the frozen SeaQuery evaluation cohort: 5,158 question-answer records referencing 1,037 original CoralVQA images. It does not include training or training-validation data.

## Files

- `images/`: original images, copied without resizing or re-encoding.
- `val_index.json`: unchanged frozen evaluation index, including its original row IDs and record order.
- `coralvqa_eval.jsonl`: the same evaluation records in JSON Lines format.
- `coralvqa_eval.csv`: portable question_id,image,question,answer index with category labels. question_id is the frozen eval_row_id.
- `coralvqa_split_imagegrouped.json`: unchanged original image split for provenance. Its training entries are not included as images. Of its 1,054 validation images, 17 have no evaluation questions and are not included.
- `source_images.json`: exact image selection and source byte sizes.
- `Build_CoralVQA_Evaluation_Subset.ipynb`: open in Colab and run all cells to finish copying images.
- `build_evaluation_subset.py`: equivalent command-line copier.
- `COPY_COMPLETE.json`: created only after every image passes a SHA-256 comparison.
- `image_manifest.csv`: source and copied Drive IDs, filenames, and original byte sizes.
- `subset_manifest.json`: counts, source hashes, and validation details.

## Attribution

Original dataset: CoralVQA, CoralReefData.
https://huggingface.co/datasets/CoralReefData/CoralVQA

This is a study-specific subset, not the complete benchmark or its official test split. The dataset retains its upstream terms and attribution. Cite the original CoralVQA dataset and the SeaQuery study when reporting results.
