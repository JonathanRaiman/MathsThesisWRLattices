#coding: utf-8

def convert_files_to_zeta

	zeta = []
	regex = /\|\| = [\d].[\d]{2}\tE_\([\d].[\d]\) = (.+)/
	regex2 = /\|\| = [\d].[\d]{2}\tE_\([\d]\) = (.+)/

	11.times do |i|
		column = []
		puts "lattice_det_1_s_#{sprintf('%.1f',2.0+0.1*i)}.txt"
		str = File.open("lattice_det_1_s_#{sprintf('%.1f',2.0+0.1*i)}.txt").read
		lines = str.split("\n")
		lines.delete_at(0)
		lines.delete_at(0)
		lines.each do |line|
			match = line.match regex
			if match
				puts match
				column << match[-1]
			else
				match = line.match regex2
				puts match
				if match
					column << match[-1]
				end
			end
		end
		zeta << column
	end
	return zeta
end

def write_zeta_to_latex zeta
	file_str = ""
	zeta[0].length.times do |j|
		row = []
		zeta.length.times do |i|
			if i>5
				row << zeta[i][j]
			end
		end
		file_str += (1.0+j*0.01).to_s + " & "+row.join(" & ")+'\\\\ \n'
	end
	File.open("output.txt", "w+") do |f|
		f << file_str
	end
end