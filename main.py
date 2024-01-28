import base64
import boto3
import cv2
import io
import json
import numpy as np
import os
from PIL import Image
import logging
from rembg import remove

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageProcessor:
    # マジックナンバーをクラス変数として定義
    THRESHOLD = 128
    GAUSSIAN_BLUR_SIZE = (1, 1)

    @staticmethod
    def create_binary_mask(masked_image):
        """マスク画像を2値化する"""
        mask = masked_image.point(lambda x: 0 if x < ImageProcessor.THRESHOLD else 255)
        return mask
    
    @staticmethod
    def remove_background(input_image_path):
        """背景を削除する"""
        try:
            input_image = Image.open(input_image_path)
        except IOError as e:
            logging.error(f"Error: Cannot open {input_image_path}, {e}")
            return None
        # rembgを使用して背景を削除
        # only_mask=True でマスク画像のみを出力
        output_image = remove(input_image,only_mask=True, alpha_matting=True)

        return output_image

    @staticmethod
    def convert_image_to_base64(image_input):
        """画像をBase64エンコードされた文字列に変換"""
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"指定されたファイルが見つかりません: {image_input}")
            with open(image_input, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")
        elif isinstance(image_input, Image.Image):
            buffer = io.BytesIO()
            image_input.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise ValueError("サポートされていない型です。str (ファイルパス) または PIL.Image.Image が必要です。")

class Translator:
    def __init__(self, region_name='us-east-1'):
        self.client = boto3.client(service_name="translate", region_name=region_name)

    def translate_text(self, text, source_language_code='ja', target_language_code='en'):
        """テキストを翻訳する"""
        try:
            result = self.client.translate_text(Text=text, SourceLanguageCode=source_language_code, TargetLanguageCode=target_language_code)
            return result.get('TranslatedText')
        except Exception as e:
            logging.error(f"翻訳中にエラーが発生しました: {e}")
            return None

class BedrockAPI:
    def __init__(self):
        self.client = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

    def invoke_model(self, body, modelId):
        """Bedrockのモデルを呼び出す"""
        response = self.client.invoke_model(body=json.dumps(body), modelId=modelId, accept="application/json", contentType="application/json")
        response_body = json.loads(response.get("body").read())
        images = [Image.open(io.BytesIO(base64.b64decode(base64_image))) for base64_image in response_body.get("images")]
        return images[0]

    def edit_image(self, task_type, prompt, negative_prompt, image, maskImage=None, seed=0):
        """画像編集タスクを実行する"""
        translator = Translator()
        translated_prompt = translator.translate_text(prompt)
        logging.info("Amazon Bedrock で画像生成を実行します。")
        logging.info(f"プロンプト（英訳前）: {prompt}")
        logging.info(f"プロンプト（英訳後）: {translated_prompt}")
        logging.info(f"ネガティブプロンプト: {negative_prompt}")
        logging.info(f"シード値: {seed}")

        body = {
            "taskType": task_type.upper(),
            "inPaintingParams": {
                "text": translated_prompt,
                "negativeText": negative_prompt,
                "image": ImageProcessor.convert_image_to_base64(image),
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "cfgScale": 8.0,
                "seed": seed,
            }
        }
        if maskImage:
            body["inPaintingParams"]["maskImage"] = ImageProcessor.convert_image_to_base64(maskImage)
        return self.invoke_model(body, modelId="amazon.titan-image-generator-v1")

def main(input_path, output_path, prompt, negative_prompt, seed=0):
    logging.info("処理を開始します。")
    
    # Rembg で背景を削除し、マスク画像を取得します
    bg_removed_image = ImageProcessor.remove_background(input_path)
    
    # マスク画像を2値化する
    mask = ImageProcessor.create_binary_mask(bg_removed_image)
    
    # マスク画像を保存する
    mask.save("./src/generated_mask.png")

    # 元画像を読み込む
    img = Image.open(input_path).convert("L")
    
    # Amazon Bedrock で画像生成を実行する
    bedrock_api = BedrockAPI()
    imageOutpaint = bedrock_api.edit_image("INPAINTING", prompt, negative_prompt, img, maskImage=mask, seed=seed)
    
    logging.info("画像の生成が完了しました。")
    imageOutpaint.save(output_path)
    logging.info(f"画像を {output_path} に保存しました。")

if __name__ == "__main__":

    # 背景を修正したい画像のパスを指定
    input_path = './src/wine.png'

    # 修正後の画像の出力パスを指定
    output_path = './src/generated_wine.png'

    # 生成したい画像の説明を指定（日本語でもOK）
    # 例: "A realistic photo of wine bottle placed on marble floors"
    # 例: "プロのカメラマンが撮影した商品画像、大理石のテーブルの上に果物がたくさん置いてある、背景は少しボケている"
    prompt = "プロのカメラマンが撮影した商品画像、大理石のテーブルの上に、たくさんの果物が置かれている、背景は少しボケている"
    
    # ネガティブプロンプトを指定
    negative_prompt = "lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, out of frame"

    main(input_path, output_path, prompt, negative_prompt)