# How to run the docker container

1. Clone the repository
2. Build the docker image
```bash
docker build -t adamchb/leaveslife:<tag> .
```
(Replace `<tag>` with your name, just as a branch in git)

3. Run the container
```bash
docker run -p 8501:8501 adamchb/leaveslife:<tag>
```

4. Go to `http://localhost:8501/`

5. You should see the jupyter notebook page, asking you for a token. The token can be found in the console where you ran the container:
```bash
  ...
  To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://a084288e7526:8501/tree?token=<token>
        http://127.0.0.1:8501/tree?token=<token>
```
Copy the token and paste it in the jupyter notebook page.

6. Run as you would normally run a jupyter notebook.

7. To stop the container, select "Close and shutdown Notebook", then "Shutdown"
