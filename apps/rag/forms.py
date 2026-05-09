from __future__ import annotations

from django import forms


class WebkitDirectoryInput(forms.FileInput):
    allow_multiple_selected = True


class RepositoryInputForm(forms.Form):
    SOURCE_TYPE_CHOICES = (
        ("zip", "Zip Upload"),
        ("folder", "Folder Upload"),
    )

    name = forms.CharField(
        max_length=255,
        required=True,
        widget=forms.TextInput(attrs={"class": "input input-bordered w-full", "placeholder": "my-repo"}),
    )
    source_type = forms.ChoiceField(
        choices=SOURCE_TYPE_CHOICES,
        required=True,
        widget=forms.Select(attrs={"class": "select select-bordered w-full"}),
    )
    zip_file = forms.FileField(
        required=False,
        widget=forms.ClearableFileInput(attrs={"class": "file-input file-input-bordered w-full", "accept": ".zip"}),
    )
    folder_files = forms.FileField(
        required=False,
        widget=WebkitDirectoryInput(
            attrs={
                "multiple": True,
                "webkitdirectory": "webkitdirectory",
                "directory": "directory",
                "class": "file-input file-input-bordered w-full",
            }
        ),
    )
    max_folder_files = 5000
    max_total_upload_bytes = 15 * 1024 * 1024  # 15 MB

    def clean(self) -> dict:
        cleaned = super().clean()
        source_type = cleaned.get("source_type")
        zip_file = cleaned.get("zip_file")
        folder_files = self.files.getlist("folder_files")

        if source_type == "zip" and not zip_file:
            self.add_error("zip_file", "Please upload a zip file.")

        if source_type == "folder" and not folder_files:
            self.add_error("folder_files", "Please choose a folder to upload.")
        if source_type == "folder" and folder_files and len(folder_files) > self.max_folder_files:
            self.add_error(
                "folder_files",
                f"Too many files selected ({len(folder_files)}). "
                f"Please select at most {self.max_folder_files} files or use a zip upload.",
            )
        if source_type == "folder" and folder_files:
            total_size = sum(f.size for f in folder_files)
            if total_size > self.max_total_upload_bytes:
                self.add_error(
                    "folder_files",
                    f"Folder upload is too large ({total_size} bytes). "
                    "Maximum allowed size is 15 MB.",
                )
        if source_type == "zip" and zip_file and zip_file.size > self.max_total_upload_bytes:
            self.add_error("zip_file", "Zip upload is too large. Maximum allowed size is 15 MB.")

        return cleaned
