# core_course_examples

## Docker

docker build -t core-course -f Dockerfile .

docker run -p 8888:8888 -v $(pwd):/cc core-course jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser