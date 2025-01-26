run:
	docker run --name raytracer-server -i -p 8080:8080 raytracer-server
	docker rm raytracer-server

build:
	docker build -t raytracer-server .

stop:
	docker stop raytracer-server
	docker rm raytracer-server