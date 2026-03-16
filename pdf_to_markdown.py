from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
from pyzerox import zerox
import sys, os
import asyncio

load_dotenv()

# 환경변수 인식 안되어서 아래 두줄 추가함
poppler_bin_path = r"C:\poppler-24.08.0\Library\bin"
os.environ["PATH"] += os.pathsep + poppler_bin_path

kwargs = {}

custom_system_prompt = None


model = "gpt-4o-mini" ## openai model


async def main():
    file_path = "./docs/income_tax.pdf" ## 문서 파일 경로
    ## process only some pages or all
    select_pages = None ## None for all, but could be int or list(int) page numbers (1 indexed)

    output_dir = "./docs" ## 결과를 저장할 경로
    result = await zerox(file_path=file_path, model=model, output_dir=output_dir,
                        custom_system_prompt=custom_system_prompt,select_pages=select_pages, **kwargs)
    return result

# encoding 문제 - py 파일로 했더니, 없어도 잘 돌아감 - 그래도 넣어둘 것!
sys.stdout.reconfigure(encoding="utf-8")  # AttributeError: 'OutStream' object has no attribute 'reconfigure' : 주피터 노트북에서는 이 에러가 남 -> py파일로 옮김
sys.stderr.reconfigure(encoding="utf-8")

# 이 부분이 주피터 노트북에서 실행이 안되는 듯 - 그래서 py파일로 따로 빼서 실행함
result = asyncio.run(main())   # 에러 -> md파일을 강의자료에서 불러와서 다음단계 진행
# result = await main()


print(result)
