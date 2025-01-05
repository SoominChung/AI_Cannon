import os
import wave
import argparse
from google.cloud import speech

import argostranslate.package
import argostranslate.translate

# Google Cloud 서비스 계정 키 파일 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service_key_team5.json"

# Google Speech-to-Text 클라이언트 초기화
client = speech.SpeechClient()


# WAV 파일의 오디오 스트림 생성
def generate_audio_stream(wav_file_path):
    with wave.open(wav_file_path, "rb") as wf:
        chunk_size = 4096  # CHUNK 크기 설정
        while True:
            data = wf.readframes(chunk_size)
            if not data:
                break
            yield speech.StreamingRecognizeRequest(audio_content=data)


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


def translate_text(text):
    translated_text = argostranslate.translate.translate(text, "ko", "en")
    print(f"Original Text  : {text}")
    print(f"Translated Text: {translated_text}")
    return translated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_file_path", type=str, default="./korean_convert.wav")
    args = parser.parse_args()
    # 입력 WAV 파일 경로
    wav_file_path = args.wav_file_path

    # WAV 파일 설정 확인
    with wave.open(wav_file_path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("WAV 파일은 모노 오디오여야 합니다.")
        if wf.getsampwidth() != 2:
            raise ValueError("WAV 파일의 샘플 크기는 16비트여야 합니다.")
        if wf.getframerate() != 44100:
            raise ValueError("WAV 파일의 샘플링 레이트는 44100Hz여야 합니다.")

    # Google Cloud Speech API 요청 설정
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,  # WAV 파일의 샘플링 레이트
        language_code="ko-KR",  # 한국어 설정
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    install_korean_to_english_package()

    # 스트리밍 요청 전송 및 응답 처리
    print("WAV 파일 음성 인식 시작...")
    try:
        responses = client.streaming_recognize(
            streaming_config, generate_audio_stream(wav_file_path)
        )
        for response in responses:
            if response.results and response.results[0].alternatives:
                # 첫 번째 결과와 대안을 출력
                transcript = response.results[0].alternatives[0].transcript
                print(f"korean text (ASR) : {transcript}")
                print(f"translated text   : {translate_text(transcript)}")

    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
