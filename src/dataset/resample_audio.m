
basedirs = {
  '../../data/badchallenge_audio/'
  '../../data/freefield1010_audio/'
  '../../data/warblr_audio/'
  }

for ix = 1:numel(basedirs)
  
  basedir = basedirs{ix};
  
  in_dir = [basedir 'wav/'];
  out_dir = [basedir 'wav_22050/'];
  
  fNames = dir([in_dir '*.wav']);
  
  if ~exist(out_dir,'dir')
    mkdir(out_dir)
  end
  
  disp(basedir)
  
  parfor jx = 1:numel(fNames)
    
    fName_out = [out_dir fNames(jx).name];
    if exist(fName_out,'file')
      continue
    end
    
    fName_in = [in_dir fNames(jx).name];
    
    [y, Fs] = audioread(fName_in);
    y = resample(y,1,2);
    audiowrite(fName_out,y,Fs/2);
    
  end
  
end


