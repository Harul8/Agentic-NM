import sys
sys.path.insert(0, r"C:\Users\rahul\Nyaymalaw-5.0")
import uvicorn
import api_server
uvicorn.run(api_server.app, host='127.0.0.1', port=8021, log_level='warning')
