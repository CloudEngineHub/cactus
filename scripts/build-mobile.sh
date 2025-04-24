# Build the React Native package
cd react 

# Create symlinks to ios and android directories if they don't exist
[ ! -e ios ] && ln -sf ../ios ios
[ ! -e android ] && ln -sf ../android android

yarn 
yarn build