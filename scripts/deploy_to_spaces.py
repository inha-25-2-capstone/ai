"""
HuggingFace Spaces 배포 스크립트
"""

import os
from huggingface_hub import HfApi, create_repo


def deploy_to_spaces():
    """HuggingFace Spaces에 앱 배포"""

    # HuggingFace 토큰 확인
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN 환경변수가 설정되지 않았습니다.")
        print("export HF_TOKEN=your_token 으로 설정해주세요.")
        return False

    # Spaces 설정
    space_id = "gaaahee/news-stance-detection"
    spaces_dir = os.path.join(os.path.dirname(__file__), "..", "spaces")

    api = HfApi(token=token)

    # Space 생성 또는 가져오기
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            token=token,
            exist_ok=True
        )
        print(f"Space created/found: {space_id}")
    except Exception as e:
        print(f"Error creating space: {e}")
        return False

    # 파일 업로드
    files_to_upload = ["app.py", "requirements.txt", "Dockerfile", "README.md"]

    for filename in files_to_upload:
        filepath = os.path.join(spaces_dir, filename)
        if os.path.exists(filepath):
            try:
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=filename,
                    repo_id=space_id,
                    repo_type="space",
                    token=token
                )
                print(f"Uploaded: {filename}")
            except Exception as e:
                print(f"Error uploading {filename}: {e}")
                return False
        else:
            print(f"File not found: {filepath}")
            return False

    print(f"\n배포 완료!")
    print(f"Space URL: https://huggingface.co/spaces/{space_id}")
    print(f"API URL: https://{space_id.replace('/', '-')}.hf.space")

    return True


if __name__ == "__main__":
    deploy_to_spaces()
