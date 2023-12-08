docker run -i -d --runtime=nvidia --name local_container \
            -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all\
                    -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v /home/zhaimingshuzms/exp/TaskAnnotator:/root/TaskAnnotator task
docker exec -it local_container bash
