"""测试环境的统一前置：把链路上报关掉。

LangSmith 一旦在 .env 里开着，跑单测就会往真实项目里塞 trace（实测发生过），
演示时要看的链路数据就被测试噪音盖掉了。测试全是离线的，这里在导入 app 之前
先把开关压成 false——dotenv 默认不覆盖已存在的环境变量，所以这样是有效的。
"""
import os

for _name in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGSMITH_TRACING_V2"):
    os.environ[_name] = "false"
for _name in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
    os.environ[_name] = ""
