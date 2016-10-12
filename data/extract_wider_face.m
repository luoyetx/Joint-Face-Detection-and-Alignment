% extarct face bbox from mat file to text file

% wider_face_train.mat
load('WIDER/wider_face_split/wider_face_train.mat', 'file_list');
load('WIDER/wider_face_split/wider_face_train.mat', 'face_bbx_list');
fout = fopen('WIDER/wider_face_split/wider_face_train.txt', 'wb');
n = size(file_list, 1); % number of events
for i = 1:n
    event = file_list{i};
    bbox = face_bbx_list{i};
    m = size(event, 1); % number of images in this event
    for j = 1:m
        face_n = size(bbox{j}, 1); % number of faces in this image
        fprintf(fout, '%s\n', event{j});
        fprintf(fout, '%d\n', face_n);
        for k = 1:face_n
            fprintf(fout, '%f %f %f %f\n', bbox{j}(k, :));
        end
    end
end
fclose(fout);

% wider_face_val.mat
load('WIDER/wider_face_split/wider_face_val.mat', 'file_list');
load('WIDER/wider_face_split/wider_face_val.mat', 'face_bbx_list');
fout = fopen('WIDER/wider_face_split/wider_face_val.txt', 'wb');
n = size(file_list, 1); % number of events
for i = 1:n
    event = file_list{i};
    bbox = face_bbx_list{i};
    m = size(event, 1); % number of images in this event
    for j = 1:m
        face_n = size(bbox{j}, 1); % number of faces in this image
        fprintf(fout, '%s\n', event{j});
        fprintf(fout, '%d\n', face_n);
        for k = 1:face_n
            fprintf(fout, '%f %f %f %f\n', bbox{j}(k, :));
        end
    end
end
fclose(fout);
