function set-direct(){
    rostopic pub -1 /auto_driver/send/direction std_msgs/Int32 "data: $1"
}

function set-speed(){
    rostopic pub -1 /auto_driver/send/speed std_msgs/Int32 "data: $1"
}

function set-mode(){
    rostopic pub -1 /auto_driver/send/mode std_msgs/Int32 "data: $1"
}

