SSH_KEY=`cat ./tomkey`
CONF=`cat ./rclone_cgis.conf`

docker build \
		--build-arg SSH_PRIVATE_KEY="$SSH_KEY" \
		--build-arg RCLONE_CONF="$CONF" \
		-t pipelineappv1 \
		--no-cache \
		.
