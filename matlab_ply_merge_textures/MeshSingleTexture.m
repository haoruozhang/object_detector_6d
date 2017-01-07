% I made this script because there was a bug in Meshlab
% when applying texture-to-vetex filter to a model with many textures,
% which is the desired format for the object detector.
% What the script does is to merge the textures to one jpg 
% and correct the coordinates accordingly.
% This is just a hacky script that works for ply that are in plain text,
% without saving the surface normals in it.
% I just include it with the object detector because I used it to convert
% the models generated from 123DCatch to plain plys that have color per vertex

plyname = 'bowl112.ply';
num_textures = 2;

im_size = 4096;
multiplier = 1;
while multiplier^2 < num_textures 
    multiplier = multiplier + 1;
end

new_texture = uint8(zeros(im_size * multiplier, im_size * multiplier, 3));


textures_completed = 0;
for i=1:multiplier
    for j=1:multiplier       
        if textures_completed >= num_textures
            break
        end              
        tex = imread(['tex_' num2str( textures_completed ) '.jpg'], 'jpeg');        
        new_texture( (i-1)*im_size + 1 : i*im_size, (j-1)*im_size +1 : j*im_size, : ) = tex;
        textures_completed = textures_completed + 1;
    end
    if textures_completed >= num_textures
        break
    end
end

mkdir('new_mesh');
imwrite(new_texture, 'new_mesh/tex_0.jpg');

ply = fopen(plyname);

new_ply = fopen(['new_mesh/' plyname], 'w+');

found = false;
while ~found
    line = fgets(ply);    
    if(length(line) > 15)
        if(strcmp(line(1:15), 'element vertex '))
            num_vertex = str2num(line(16:end));
            found = true;
        end
    end
    fprintf(new_ply, '%s', line);
end

found = false;
while ~found
    line = fgets(ply);    
    if(length(line) > 13)
        if(strcmp(line(1:13), 'element face '))
            num_face = str2num(line(14:end));
            found = true;
        end
    end
    fprintf(new_ply, '%s', line);
end

found = false;
while ~found
    line = fgets(ply);    
    if(length(line) >= 10)
        if(strcmp(line(1:10), 'end_header'))            
            found = true;
        end
    end
    fprintf(new_ply, '%s', line);
end

for i=1:num_vertex
    line = fgets(ply);
    fprintf(new_ply, '%s', line);
end

for i=1:num_face
    line = fgets(ply);
    data = str2num(line);
    texture_num = data(12);
    if(texture_num ~= -1)
        col = mod(texture_num, multiplier);
        row = floor(texture_num/multiplier);
        offset_row = (multiplier - row - 1)* (1/multiplier);
        offset_col = (col)* (1/multiplier);
        data([6 8 10]) = offset_col + data([6 8 10]) / multiplier;
        data([7 9 11]) = offset_row + data([7 9 11]) / multiplier;
        data(12) = 0;
    end    
    for j=1:5
        fprintf(new_ply,'%d ',data(j));
    end
    for j=6:11
        fprintf(new_ply,'%f ',data(j));
    end
    for j=12:16
        fprintf(new_ply,'%d ',data(j));
    end
    fprintf(new_ply,'\n');
    
end
    

fclose(ply);
fclose(new_ply);
    


    
