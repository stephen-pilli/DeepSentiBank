function features = read_features(name,layer)

if ~exist('name','var') || ~exist('layer','var')
    disp('Running with default arguments: read_features(''test_image-features'', ''prob'').');
    disp('layer can be ''fc6'', ''fc7'', ''fc8-t'', ''prob''.');
    name = 'test_image-features';
    layer = 'prob';
end

dim=4096;
if strcmp(layer,'fc8-t') || strcmp(layer,'prob')
    dim=2089;
end
disp(['load features for layer ' layer ' of ' name ' ...']);

imagelist = importdata([name '.txt']);
imagelist = imagelist.textdata;
ins_num = length(imagelist);
f = fopen([name '_' layer '.dat']);
features = fread(f,[dim ins_num],'single');
fclose(f);

disp([num2str(ins_num) ' ' num2str(dim) '-dimension features are loaded!']);
end