docker run -i -d -p 5000:5000 -p 5001:5001 --runtime=nvidia --name task_container \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
                    -v ~/:/root/hza_home \
                    task
                                    
docker exec -it task_container bash
# -e QT_X11_NO_MITSHM=1 -e XAUTHORITY 
