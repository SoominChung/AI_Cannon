import time
from transformers import MarianMTModel, MarianTokenizer
import argostranslate.package
import argostranslate.translate


def translate_text(text):
    model_name = "Helsinki-NLP/opus-mt-ko-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(f"Original Text  : {text}")
    print(f"Translated Text: {translated_text}")
    return translated_text


def install_korean_to_english_package():
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # 한국어에서 영어로의 패키지 필터링
    ko_en_packages = [
        pkg
        for pkg in available_packages
        if pkg.from_code == "ko" and pkg.to_code == "en"
    ]

    if not ko_en_packages:
        print("No ko-en packages found")
        return

    package_to_install = ko_en_packages[0]
    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)
    print("ko-en package installed")


def translate_text2(text):
    translated_text = argostranslate.translate.translate(text, "ko", "en")
    print(f"Original Text  : {text}")
    print(f"Translated Text: {translated_text}")
    return translated_text


# 실행 부분
if __name__ == "__main__":
    # Helsinki-NLP 모델 테스트
    korean_texts = [
        "발사해",
        "가장 가까운 좀비를 쏴",
        "오른쪽의 좀비를 조준해",
        "다 쏴버려",
    ]
    for text in korean_texts:
        start_time = time.time()
        translate_text(text)
        end_time = time.time()
        print(f"Execution Time: {end_time - start_time:.2f} seconds\n")

    # Argos Translate 설치 및 테스트
    install_korean_to_english_package()
    for text in korean_texts:
        start_time = time.time()
        translate_text2(text)
        end_time = time.time()
        print(f"Execution Time: {end_time - start_time:.2f} seconds\n")
