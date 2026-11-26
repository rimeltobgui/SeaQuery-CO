# Import the CoralVQA evaluation folder into SeaQuery-CO

The manual GitHub Actions workflow downloads the completed evaluation folder from Drive into `data/datasets/coralvqa/` and commits it to the selected branch. It preserves individual images and files; no archive is created. Images use Git LFS. The frozen subset contains 1,037 images, 5,158 question-answer records, and 2,128,477,353 image bytes.

## One-time private Drive access

1. In [Google Cloud](https://console.cloud.google.com/), select or create a project and enable the **Google Drive API**.
2. Under **IAM & Admin → Service Accounts**, create a service account. It does not need a project IAM role for reading this shared Drive folder.
3. Open the service account, select **Keys → Add key → Create new key → JSON**, and download the key.
4. Copy the key's `client_email`. Share only the [evaluation folder](https://drive.google.com/drive/folders/1GzQGfcq0N4cb8J1jlAyS9P2YDqJZHp78) with that email as **Viewer**. The images and metadata must inherit its access. The folder can remain private.
5. In **SeaQuery-CO → Settings → Secrets and variables → Actions → New repository secret**, create **GDRIVE_SERVICE_ACCOUNT_JSON**. Paste the entire downloaded JSON key into the secret value. Do not add the key to the repository or chat.

## Run

Open **Actions → Import CoralVQA from Drive → Run workflow**. Select `main`, or another branch containing both the workflow and downloader, then run it. The workflow writes to that branch using the repository's built-in `GITHUB_TOKEN`; no GitHub personal access token is required. Branch rules still apply. A rejected push leaves the existing remote branch unchanged; rerun after resolving the reported permissions or branch-rule issue.

The workflow downloads metadata and images into temporary storage first. It verifies the pinned evaluation-index and image-manifest hashes, annotation consistency, all image sizes, and every image SHA-256 before copying into the repository and committing. It refuses duplicate Drive filenames, incomplete folders, or mismatched files. Repeat runs do not create a new commit when the imported files are unchanged. They still download the selected subset again for verification.

The workflow adds `data/datasets/coralvqa/images/**` to `.gitattributes` while preserving its existing rules. Existing Coralscapes and adapter LFS objects are not downloaded by checkout. Git LFS storage and download usage count toward the repository owner's allowance.

To get actual images in an existing local checkout after the import:

```bash
git pull
git lfs pull --include='data/datasets/coralvqa/images/**'
```

GitHub source ZIP downloads may contain LFS pointers unless the repository's archive setting includes LFS objects. For the Code Ocean upload, use the downloaded image files, not LFS pointers.

## Files

- `.github/workflows/import-coralvqa.yml`
- `code/scripts/import_coralvqa_from_drive.py`

Validation before publishing: local fixtures cover complete imports, repeat imports, corrupted images, missing images, changed indices, duplicate names, and paginated listings. The actual completed subset's metadata also passed validation. Full GitHub-hosted execution requires the Drive credential above and has not yet been run.

## References

- [Drive file downloads](https://developers.google.com/workspace/drive/api/guides/manage-downloads)
- [Service-account JSON keys](https://docs.cloud.google.com/iam/docs/keys-create-delete)
- [GitHub Actions repository secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets)
- [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Git LFS usage](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
