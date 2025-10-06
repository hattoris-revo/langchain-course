from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_aws import ChatBedrockConverse

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    イーロン・リーヴ・マスク（英語: Elon Reeve Musk、1971年6月28日 - ）は、南アフリカ共和国出身のアメリカ合衆国の起業家[3]。2025年以降、第2次トランプ政権下で、公式には特別政府職員の職位で大統領であるドナルド・トランプの大統領上級政治顧問の役割を務め[4][5][6][7]、政府効率化省（DOGE）の事実上のトップとして活動した[8][注 1]。PayPal[注 2]、スペースX、テスラ[注 3]、ボーリング・カンパニー、OpenAI、xAI等を共同設立し[12][13]、スペースX、テスラのCEO、X社（旧：Twitter[14]）の執行会長兼CTOを務めた[15][16]。南アフリカ共和国、カナダ、アメリカ合衆国の国籍を持つ。

    電子決済の先駆的企業PayPalの創業者の一人として成功を収め[17]、その後電気自動車（テスラ）、宇宙開発（スペースX）、太陽光発電などのビジネスでも成功し、当時没落していたそれらの業界を再興させたと評される[18][19]。ピーター・ティールやYouTube創業者のチャド・ハーリーなどと共に「ペイパルマフィア」の一人としても語られる[20][21][22]。

    2012年にはスペースXが国際宇宙ステーションへの宇宙船の打ち上げに成功し、テスラはEV ｢モデルS｣ を発売した。これによってマスクは宇宙と自動車というまったく別の業界で偉業を成し遂げ、スティーブ・ジョブズに例えられる存在になった[23]。

    2025年3月の時点でマスクは世界で最も裕福な人物であり、その純資産は3420億ドル（約51兆200億円）と見積もられている[24]。マスクの資産は2024年12月には一時4470億ドル（約68兆円）に達しており、これは世界で初めて個人資産が4000億ドルを超えた事例となった[25]。（詳細は資産を参照）

    2019年にフォーブスが発表した「アメリカ合衆国で最も革新的なリーダー」ランキングではAmazon.com（アマゾン）CEOのジェフ・ベゾスと並び、第1位の評価を受けた[26]。また、自身の弟が起業したテスラの子会社ソーラーシティの会長を務めている[27]。

    2025年1月から第47代米大統領に就任したドナルド・トランプが設置した政府効率化省（DOGE）のトップを務めており[28][29]、その影響力からTIME紙は「米国政府の構造に対し、これほどまでに権力を振るった一市民は前例がない」としており[30]、「影の大統領」とも評されていた[31][32][33]。また、スペースXの企業城下町でテキサス州の自治体となったスターベース市（英語版）の「影の市長」とも呼ばれた[34]。

    一方、同年2月17日司法省は、ワシントン連邦地裁のターニャ・チャトカン判事に提出した宣誓供述書で、マスクは「DOGEの責任者ではなく、大統領の上級顧問」だと述べた[5]。供述書では、マスクはDOGEで働いているのではなく、大統領に助言および指示を伝えるために動いているにすぎず、「政府の意思決定を行う実際の権限も正式な権限も持っていない」と記している[6]。同年5月28日、特別政府職員としての任期が終了するとして、政府の役職を離れると発表した[11][35][36]。
    """

    summary_template = """
    given the imformation {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["imformation"], template=summary_template
    )

    llm = ChatBedrockConverse(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        # region_name=...,
        # aws_access_key_id=...,
        # aws_secret_access_key=...,
        # aws_session_token=...,
        temperature=0,
        # max_tokens=...,
        # other params...
    )
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
