$task = aws ecs list-tasks --cluster agentmind-cluster --region eu-central-1 --query "taskArns[0]" --output text
$eni = aws ecs describe-tasks --cluster agentmind-cluster --tasks $task --region eu-central-1 --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text
$ip = aws ec2 describe-network-interfaces --network-interface-ids $eni --query "NetworkInterfaces[0].Association.PublicIp" --output text --region eu-central-1
Write-Host "AgentMind live at: http://$ip`:8000/health"
