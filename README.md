Backend --install /projects
check for python version. if not, download python latest.

- pip3 install openai fastapi uvicorn python-multipart
- pip3 install -r requirements.txt
- python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

After running this, in your command prompt 

In Windows
- ipconfig

check for iP4 address copy and paste in frontend, backend url

In Mac
- ipconfig getifaddr en0;
