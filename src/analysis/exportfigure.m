%exportfigure(handle,filename,PaperSize,DPI)
% PaperSize: in figure units (usually inches)
% DPI: resolution at which to generate figure
function exportfigure(handle,filename,PaperSize,DPI)

ext=filename(end-3:end);

% default handle should be gcf
if ~exist('handle', 'var')
    handle=gcf;
end


set(handle,'PaperSize', PaperSize);
set(handle,'PaperPosition', [0 0 PaperSize]);
if strcmpi(ext,'.jpg')
    print(handle,'-djpeg100', ['-r' num2str(DPI)], filename);
elseif strcmpi(ext,'.eps')
    print(handle,'-depsc', filename);
elseif strcmpi(ext,'.png')
    print(handle,'-dpng', ['-r' num2str(DPI) ' -painters ' ], filename);
elseif strcmpi(ext,'.tif')
    print(handle,'-dtiffn', filename);
elseif strcmpi(ext,'.pdf')
%   saveas(handle, filename);
    print(handle,'-dpdf', filename);
else
    error('File extention not handled yet.');
end