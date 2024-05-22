<h1 align="center">Assignment 7 Big Data Laboratory</h1>

## Usage

You'll need to run the docker containers:

```bash
docker-compose up
```

This spins up 3 containers in their respective ports:

- Prometheus: http://localhost:9090/
- Grafana: http://localhost:3000/
- FastAPI: http://localhost:8000/

On the FastAPI, you can access `/metrics` endpoint to see the data Prometheus is scraping from it.
