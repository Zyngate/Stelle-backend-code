
##  Deploying a FastAPI App to Amazon EC2: A Step-by-Step Guide

This guide will walk you through launching an AWS EC2 instance, setting up the environment, and deploying your application using Gunicorn as a process manager and Nginx as a reverse proxy.

### Prerequisites

  * An **AWS Account**.
  * Your application code pushed to a **Git repository**.
  * An SSH client installed on your local machine.

-----

### Step 1: Launch an EC2 Instance

1.  **Navigate to the EC2 Dashboard:** Log in to your AWS Console and go to the EC2 service.
2.  **Launch Instance:** Click the "Launch instances" button.
3.  **Choose an AMI:** Select an Amazon Machine Image (AMI). **Ubuntu Server (LTS)** is a great choice.
4.  **Select an Instance Type:** For a small to medium application, a `t3.small` or `t2.micro` (for testing) is a good starting point.
5.  **Create a Key Pair:** In the "Key pair (login)" section, create a new key pair. Give it a name (e.g., `stelle-app-key`), select `.pem` format, and download the file. **Keep this file safe; you can't download it again.**
6.  **Configure Security Group:** This is crucial for allowing traffic to your app. In "Network settings", click "Edit".
      * Keep the default SSH rule (Port 22).
      * Click "Add security group rule" and add the following:
          * **Type:** `HTTP`, **Port range:** `80`, **Source:** `Anywhere` (`0.0.0.0/0`)
          * **Type:** `HTTPS`, **Port range:** `443`, **Source:** `Anywhere` (`0.0.0.0/0`)
          * **Type:** `Custom TCP`, **Port range:** `8000`, **Source:** `Anywhere` (`0.0.0.0/0`) (This allows direct testing of your app).
7.  **Launch:** Review the settings and click "Launch instance".

-----

### Step 2: Connect to Your EC2 Instance

1.  Find your instance's **Public IPv4 address** from the EC2 dashboard.
2.  Open a terminal on your local machine.
3.  Use the `ssh` command to connect. Make sure your `.pem` key has the correct permissions.

<!-- end list -->

```bash
# Set correct permissions for your key file
chmod 400 /path/to/your/stelle-app-key.pem

# Connect to the instance (replace with your details)
ssh -i /path/to/your/stelle-app-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

-----

### Step 3: Set Up the Server Environment

Once connected, run these commands to prepare your server.

```bash
# Update and upgrade system packages
sudo apt update && sudo apt upgrade -y

# Install Python, pip, and the virtual environment module
sudo apt install python3-pip python3-venv -y

# Install Git to clone your code
sudo apt install git -y
```

-----

### Step 4: Clone and Prepare Your Application


2.  **Create a Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Create `requirements.txt`:** On your local machine, create a file named `requirements.txt` with all the necessary packages from your `main17.py` file. It should look something like this:

    ```text
    # requirements.txt
    fastapi
    uvicorn
    motor
    python-dotenv
    pywebpush
    docx2txt
    PyMuPDF
    loguru
    ratelimit
    sentence-transformers
    faiss-cpu # Use faiss-gpu if you have a GPU instance
    groq
    pydantic
    httpx
    numpy
    pytz
    html2image
    python-multipart
    gunicorn
    ```

    Commit and push this file to your repository, then pull the changes on your EC2 instance (`git pull`).

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

-----

### Step 5: Configure Environment Variables

Your application relies on a `.env` file for sensitive keys. Create and populate this file on the server.

```bash
# Use a terminal editor like nano to create the file
nano .env
```

Paste your environment variables into the editor. It should look like this:

```ini
# .env file
MONGO_URI="your_mongodb_connection_string"
VAPID_PUBLIC_KEY="your_public_key"
VAPID_PRIVATE_KEY="your_private_key"
SMTP_USERNAME="your_email@example.com"
SMTP_PASSWORD="your_email_password"

# Groq API Keys
GROQ_API_KEY_="..."
GROQ_API_KEY_GENERATE_1="..."
GROQ_API_KEY_GENERATE_2="..."
# ... and all other GROQ keys ...
```

Save the file by pressing `CTRL + X`, then `Y`, then `Enter`.

-----

### Step 6: Run and Test with Uvicorn

Before setting up the production servers, make sure the app runs correctly.

```bash
# Make sure your virtual environment is active
# (venv) $
uvicorn main17:app --host 0.0.0.0 --port 8000
```

Open your browser and navigate to `http://YOUR_EC2_PUBLIC_IP:8000/docs`. You should see the FastAPI documentation. If it works, stop the server with `CTRL + C`.

-----

### Step 7: Set Up Gunicorn and Nginx for Production

We will use **Gunicorn** to manage the FastAPI application and **Nginx** as a reverse proxy to handle incoming web traffic and forward it to Gunicorn.

#### A. Configure Gunicorn

Run your app with Gunicorn to ensure it works. The `-k uvicorn.workers.UvicornWorker` flag tells Gunicorn to use Uvicorn's worker class, which is necessary for `asyncio` applications like FastAPI.

```bash
# (venv) $
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main17:app -b 0.0.0.0:8000
```

This command starts Gunicorn with 4 worker processes. Test it again at `http://YOUR_EC2_PUBLIC_IP:8000/docs`.

#### B. Install and Configure Nginx

1.  **Install Nginx:**

    ```bash
    sudo apt install nginx -y
    ```

2.  **Create an Nginx Configuration File:**

    ```bash
    sudo nano /etc/nginx/sites-available/fastapi_app
    ```




### Step 8: Run the App as a Systemd Service

To ensure your application runs continuously and restarts automatically on server reboots or crashes, we'll create a `systemd` service file.

1.  **Create the Service File:**

    ```bash
    sudo nano /etc/systemd/system/fastapi_app.service
    ```

2.  **Paste the following configuration.** Be sure to replace `/home/ubuntu/your-app` with the actual path to your project directory.

    ```ini
    [Unit]
    Description=Gunicorn instance to serve FastAPI app
    After=network.target

    [Service]
    User=ubuntu
    Group=www-data
    WorkingDirectory=/home/ubuntu/your-app
    Environment="PATH=/home/ubuntu/your-app/venv/bin"
    ExecStart=/home/ubuntu/your-app/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main17:app -b 0.0.0.0:8000

    [Install]
    WantedBy=multi-user.target
    ```

3.  **Start and Enable the Service:**

    ```bash
    # Reload the systemd daemon to recognize the new service
    sudo systemctl daemon-reload

    # Start your FastAPI application service
    sudo systemctl start fastapi_app

    # Enable the service to start automatically on boot
    sudo systemctl enable fastapi_app
    ```

4.  **Check the Status:**

    ```bash
    sudo systemctl status fastapi_app
    ```

    You should see an "active (running)" status.

