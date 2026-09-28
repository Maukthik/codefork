from dotenv import load_dotenv
load_dotenv()
from contree_sdk import ContreeSync

client = ContreeSync()
sandbox = client.images.use("python:3.12-slim")
result = sandbox.run(shell="python -c \"print('hello from nebius sandbox')\"").wait()
print(result.exit_code, result.stdout)