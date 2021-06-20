A small flask app to manage post-processing tasks with chupiflow.
Not a professional web-app. Consider setting it up with `nginx` https login.

## Running

A service can be started in `/etc/systemd/system/chupiflow.service` using `gunicorn`. Follow [this instructional](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04) to set it up. Note that everything is stored in `/home/smartcitizen-data` and that necessary tokens are stored in a `.env` file.

```
[Unit]
Description=Gunicorn instance to serve chupiflow-ui
After=network.target

[Service]
User=root
Group=www-data
EnvironmentFile=-/home/smartcitizen-data/.env
WorkingDirectory=/home/smartcitizen-data/tasks/chupiflow_ui
ExecStart=/usr/local/bin/gunicorn --workers 3 --bind unix:chupiflow.sock -m 007 wsgi:app

[Install]
WantedBy=multi-user.target
```

Normal service routines apply:

```
systemctl start chupiflow.service
systemctl enable chupiflow.service
```