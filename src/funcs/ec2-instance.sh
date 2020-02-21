#! /bin/bash

cd $( dirname $0 )

id=i-0bafe58784e318755

case $1 in
    "rsync")
        pubip=$( ./ec2-instance.sh pubip )

        rsync -zz -auzh -e "ssh -i /home/thorpedoes/.ssh/id_rsa" ../funcs ../common jothor@${pubip}:func-testing/
        ;;
    "ssh")
        pubip=$( ./ec2-instance.sh pubip )

        ssh -i /home/thorpedoes/.ssh/id_rsa jothor@${pubip}
        ;;
    "pubip")
        aws ec2 describe-instances --filter Name=instance-id,Values=${id} --query Reservations[*].Instances[*].PublicIpAddress --output text
        ;;
    "start")
        aws ec2 start-instances --instance-ids ${id}
        ;;
    "stop")
        aws ec2 stop-instances --instance-ids ${id}
        ;;
    *)
        echo "Unrecognzied option"
        ;;
esac
