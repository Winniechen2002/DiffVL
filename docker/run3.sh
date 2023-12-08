docker run -i -d -p 5000:5000 -p 5001:5001 --runtime=nvidia --name container3 \
            -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all\
                    -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v /home/zhaimingshuzms/exp/TaskAnnotator:/root/TaskAnnotator task
docker exec -it container3 bash
